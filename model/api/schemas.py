from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PredictionInput(BaseModel):
    date: str
    month: int
    기온: float
    강수량: float
    풍속: float
    구이름: str

class PredictionOutput(BaseModel):
    date: str
    pm10: float = Field(..., description="예측된 미세먼지 농도", examples=[67.22])

# 여러 날짜별 예측을 위한 입력/출력 스키마
class BundleInput(BaseModel):
    data: List[PredictionInput]

class BundleOutput(BaseModel):
    data: List[PredictionOutput] 