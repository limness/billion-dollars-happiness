from pydantic_settings import BaseSettings
from pydantic import Field


class BillionSettings(BaseSettings):
    EXTR_WINDOW: int = Field('5')
    PATTERN_SIZE: int = Field('20')
    UNTIL_PROFIT: int = Field('32')
    PROFIT_VALUE: float = Field('0.8')
    EMBEDDING_DIM: int = Field('100')
    BATCH_SIZE: int = Field('150')
    MARGIN: int = Field('1')
    EPOCHS: int = Field('32')
    LEARNING_RATE: float = Field('0.00009470240447408595')


billion_settings = BillionSettings()
