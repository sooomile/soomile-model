from pydantic import BaseModel, Field
from typing import List

class PredictionInput(BaseModel):
    #station_name: str = Field(..., description="측정소명", examples=["강남구"])
    pm25: float = Field(..., description="초미세먼지 농도", examples=[25.0])
    기온: float = Field(..., description="평균기온(°C)", examples=[5.2])
    강수량: float = Field(..., description="일강수량(mm)", examples=[1.2])
    풍속: float = Field(..., description="평균 풍속(m/s)", examples=[2.1])
    month: int = Field(..., description="월", examples=[3])

class PredictionOutput(BaseModel):
    #station_name: str = Field(..., description="측정소명", examples=["강남구"])
    pm10: float = Field(..., description="예측된 미세먼지 농도", examples=[67.22]) 