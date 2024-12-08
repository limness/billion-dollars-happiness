from pydantic import BaseModel, Field


class PredictRequestSchema(BaseModel):
    # time: list = Field(description="")
    xau_open: list = Field(description="XAUUSD Open Feature")
    xau_high: list = Field(description="XAUUSD High Feature")
    xau_low: list = Field(description="XAUUSD Low Feature")
    xau_close: list = Field(description="XAUUSD Close Feature")
    xau_volume: list = Field(description="XAUUSD Volume Feature")
    gvz_open: list = Field(description="CBOE Gold Volatitity Open Feature")
    gvz_high: list = Field(description="CBOE Gold Volatitity High Feature")
    gvz_low: list = Field(description="CBOE Gold Volatitity Low Feature")
    gvz_close: list = Field(description="CBOE Gold Volatitity Close Feature")
    tnx_open: list = Field(description="10 Year Treasury Yield Open Feature")
    tnx_high: list = Field(description="10 Year Treasury Yield High Feature")
    tnx_low: list = Field(description="10 Year Treasury Yield Low Feature")
    tnx_close: list = Field(description="10 Year Treasury Yield Close Feature")
    dxy_open: list = Field(description="U.S. Dollar Index Open Feature")
    dxy_high: list = Field(description="U.S. Dollar Index High Feature")
    dxy_low: list = Field(description="U.S. Dollar Index Low Feature")
    dxy_close: list = Field(description="U.S. Dollar Index Close Feature")


class PredictResponseSchema(BaseModel):
    signal: int = Field(description="")
