import joblib
import pandas as pd
from schemas import PredictionInput, PredictionOutput # schemas.py 파일에서 입출력 형식을 가져옵니다.

# --- 중요 ---
# 노트북에서 PM2.5를 예측하도록 새로 학습시킨 모델의 경로입니다.
MODEL_PATH = "../models/pm10_predict_model.pkl" 

class ModelService:
    _instance = None
    
    # 싱글턴 패턴: 앱 전체에서 모델 객체를 한 번만 생성하고 공유합니다.
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # 모델이 이미 로드되었는지 확인하여, 중복 로드를 방지합니다.
        if not hasattr(self, 'model'):
            self.model = self._load_model()
            # 모델 학습에 사용된 특성들의 이름과 순서입니다.
            # 이 순서는 반드시 모델 학습 때 사용된 순서와 일치해야 합니다.
            # 터미널 디버깅 결과에 따라 순서를 수정합니다.
            self.feature_columns = [
                'pm25',
                '평균기온(°C)',
                '일강수량(mm)',
                '평균 풍속(m/s)',
                'month'
            ]

    def _load_model(self):
        """지정된 경로에서 모델 파일을 로드합니다."""
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ 모델 로드 성공: {MODEL_PATH}")

            # --- 🕵️‍♂️ 디버깅 코드 추가 ---
            if hasattr(model, 'feature_names_in_'):
                print(f"👀 [디버깅] 로드된 모델의 학습 특성: {model.feature_names_in_}")
            else:
                print("👀 [디버깅] 이 모델에는 feature_names_in_ 속성이 없습니다.")
            # --- 디버깅 코드 끝 ---
            
            return model
        except FileNotFoundError:
            print(f"[에러] 모델 파일을 찾을 수 없습니다. 경로: {MODEL_PATH}")
            print("[해결] 노트북에서 초미세먼지 예측 모델을 학습시키고, 위 경로에 맞게 저장했는지 확인하세요.")
            return None
        except Exception as e:
            print(f"[에러] 모델 로드 중 예상치 못한 오류가 발생했습니다: {e}")
            return None

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """단일 입력 데이터를 받아 초미세먼지 농도를 예측합니다."""
        if not self.model:
            # 모델이 로드되지 않았을 경우, 에러를 발생시켜 문제를 명확히 알립니다.
            raise RuntimeError("모델이 정상적으로 로드되지 않았습니다. 서버 시작 로그를 확인해주세요.")

        # 1. API로 받은 입력 데이터를 Pandas DataFrame으로 변환합니다.
        #    입력 데이터가 하나이므로, 리스트로 감싸서 DataFrame을 생성합니다.
        input_df = pd.DataFrame([input_data.dict()])
        
        # 2. API에서 사용한 영문 컬럼명을 모델이 이해하는 한글 컬럼명으로 변경합니다.
        predict_df = input_df.rename(columns={
            "기온": "평균기온(°C)",
            "강수량": "일강수량(mm)",
            "풍속": "평균 풍속(m/s)"
        })

        # 3. 모델 학습에 사용된 순서와 이름에 맞게 특성(X) 데이터를 준비합니다.
        X = predict_df[self.feature_columns]

        # 4. 모델을 사용하여 예측을 수행합니다.
        prediction = self.model.predict(X)
        predicted_value = prediction[0] # 예측 결과는 배열이므로 첫 번째 값을 사용합니다.

        # 5. 최종 결과를 API 출력 형식에 맞게 생성하여 반환합니다.
        result = PredictionOutput(
            #station_name=input_data.station_name,
            predicted_pm10=round(predicted_value, 2) # 소수점 2자리까지만 반올림
        )
        
        return result

# 앱 전체에서 사용할 모델 서비스 인스턴스를 생성합니다.
# FastAPI 앱이 시작될 때 이 코드가 실행되면서 모델이 메모리에 로드됩니다.
model_service = ModelService()