from fastapi import FastAPI, HTTPException
from typing import List

# 우리가 만든 스키마(데이터 형식)와 모델 서비스를 가져옵니다.
from schemas import PredictionInput, PredictionOutput, BundleInput, BundleOutput
from model_service import model_service

# FastAPI 앱 생성
app = FastAPI(
    title="미세먼지 예측 API",
    description="PM2.5과 날씨 정보를 기반으로 PM10를 예측하는 API입니다.",
    version="1.0.0"
)

# 서버가 잘 켜졌는지 확인하는 기본 경로
@app.get("/")
def read_root():
    return {"message": "✅ 미세먼지 예측 API 서버가 정상적으로 실행 중입니다."}

# 예측을 수행하는 메인 엔드포인트
# 이 주소로 POST 요청을 보내면 예측을 수행합니다.
@app.post("/predict", response_model=PredictionOutput)
def predict_pm25(data: PredictionInput):
    """
    단일 측정소의 초미세먼지(PM25) 및 날씨 데이터를 입력받아,
    해당 측정소의 미세먼지(PM10)를 예측합니다.
    """
    try:
        # 모델 서비스의 predict 함수를 호출하여 예측을 수행합니다.
        result = model_service.predict(data)
        return result
    except RuntimeError as e:
        # 모델이 로드되지 않았을 경우, 503 에러를 반환합니다.
        raise HTTPException(status_code=503, detail=f"서비스를 사용할 수 없습니다: {e}")
    except Exception as e:
        # 그 외 예측 중 에러가 발생하면 500 에러를 반환합니다.
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {e}")

@app.post("/predict/bundle", response_model=BundleOutput)
def predict_bundle(bundle_input: BundleInput):
    """
    여러 날짜별로 구, 날짜, 기상정보를 받아 date/pm10만 리스트로 반환합니다.
    """
    try:
        result = model_service.predict_bundle(bundle_input)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"서비스를 사용할 수 없습니다: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {e}") 

# 서버 시작 시 모델이 로드되었는지 확인하는 로직 (선택사항이지만 추천)
@app.on_event("startup")
async def startup_event():
    if model_service.model is None:
        print("🚨 [경고] 서버가 시작되었지만, 모델이 로드되지 않았습니다! API가 정상 동작하지 않을 수 있습니다.")
    else:
        print("🚀 FastAPI 애플리케이션 시작 준비 완료. 모델이 성공적으로 로드되었습니다.")

