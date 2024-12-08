import pandas as pd
import torch
from fastapi import status, HTTPException, UploadFile, FastAPI, File, Body
from loguru import logger

from app.core.settings import janus_settings
from server.schemas import PredictResponseSchema
from app.helpers import featurized
from app.net import NetworkDriver
from app import net

logger.add("logs/outputs.txt", level="DEBUG", backtrace=True, diagnose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Application Base
app = FastAPI(
    description="Tools for working with the application catalog via the HTTP REST protocol",
)


@app.post(
    "/models/replace",
    status_code=status.HTTP_200_OK,
    response_description="Get all applications",
    response_model=PredictResponseSchema,
)
async def replace(
    anchor_no_pattern: UploadFile = File(None),
    anchor_pattern: UploadFile = File(None),
    model: UploadFile = File(None),
    scaler: UploadFile = File(None),
) -> dict:
    """
    Replaces janus model on new files
    Args:
        anchor_no_pattern: New Anchor No Pattern file
        anchor_pattern: New Anchor Pattern file
        model: New Model file
        scaler: New Scaler file

    Returns:
        Dictionary including signal
    """
    if not any([anchor_no_pattern, anchor_pattern, model, scaler]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                'status': 'error',
                'details': 'At least one file must be specified!',
            },
        )

    # todo: add file extension checks

    if anchor_no_pattern is not None:
        with open("app/model/anchor_no_pattern.pt", "wb+") as file_object:
            file_object.write(anchor_no_pattern.file.read())

    if anchor_pattern is not None:
        with open("app/model/anchor_pattern.pt", "wb+") as file_object:
            file_object.write(anchor_pattern.file.read())

    if model is not None:
        with open("app/model/model.pt", "wb+") as file_object:
            file_object.write(model.file.read())

    if scaler is not None:
        with open("app/model/scaler.save", "wb+") as file_object:
            file_object.write(scaler.file.read())

    # todo: make method for replacing data directly inside NetworkDriver
    #  and remove `from app import net`
    net.net_driver = NetworkDriver(pattern_size=30, embedding_dim=30 * 10)

    return {}


@app.post(
    "/models/predict",
    status_code=status.HTTP_200_OK,
    response_description="Get all applications",
    response_model=PredictResponseSchema,
)
async def predict(data: dict = Body()) -> dict:
    """
    Getting a neuro prediction signal for market stock
    Args:
        data: Basically XAU, GVZ, TNX, DXY

    Returns:
        Dictionary including signal
    """
    # create window
    df = pd.DataFrame.from_dict(data)
    dataset = df.values[-janus_settings.EXTR_WINDOW :, 3]

    # if the current point is not an extremum in the exr_window,
    # then no prediction is required
    if dataset[-1] != max(dataset) and dataset[-1] != min(dataset):
        return {'signal': 7}

    dataset_featured = featurized(df)
    data = net.net_driver.scaler.transform(dataset_featured.values)
    tensor = torch.from_numpy(data[None, None, ...]).type(torch.float32).to(device)
    signal = net.net_driver.predict(tensor)

    # if the model says it is a pattern, look at extremum
    if dataset[-1] == max(dataset) and signal != 0:
        signal = -1

    elif dataset[-1] == min(dataset) and signal != 0:
        signal = 1

    return {'signal': signal}
