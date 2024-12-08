import pandas as pd
import torch
import numpy as np
from fastapi import status, FastAPI, Body
from loguru import logger

from server.schemas import PredictResponseSchema, PredictRequestSchema
from app.helpers import featurized
from app.net import net_driver

logger.add("logs/outputs.txt", level="DEBUG", backtrace=True, diagnose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Application Base
app = FastAPI(
    # title=application_settings.APP_TITLE,
    # debug=application_settings.APP_DEBUG,
    # version=application_settings.APP_VERSION,
    description="Tools for working with the application catalog via the HTTP REST protocol",
)


@app.post(
    "/models/predict",
    status_code=status.HTTP_200_OK,
    response_description="Get all applications",
    response_model=PredictResponseSchema,
)
async def predict(data: PredictRequestSchema = Body()) -> dict:
    print(data)
    df = pd.DataFrame.from_dict(dict(data))
    df = featurized(df)
    print(df)
    data = net_driver.scaler.transform(df.values.tolist())
    # data = torch.from_numpy(data).type(torch.float32)
    tensor = torch.from_numpy(data[None, None, ...]).type(torch.float32).to(device)

    return {'signal': net_driver.predict(tensor)}
