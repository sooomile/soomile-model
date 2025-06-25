import joblib
import pandas as pd
from schemas import PredictionInput, PredictionOutput, BundleInput, BundleOutput # schemas.py 파일에서 입출력 형식을 가져옵니다.
import os
from dotenv import load_dotenv
import requests

# --- 중요 ---
# 노트북에서 PM2.5를 예측하도록 새로 학습시킨 모델의 경로입니다.
MODEL_PATH = "../models/pm10_predict_model.pkl" 

# 서울시 25개 구 한글-영문 매핑
KOR_TO_ENG_GU = {
    "강남구": "Gangnam-gu",
    "강동구": "Gangdong-gu",
    "강북구": "Gangbuk-gu",
    "강서구": "Gangseo-gu",
    "관악구": "Gwanak-gu",
    "광진구": "Gwangjin-gu",
    "구로구": "Guro-gu",
    "금천구": "Geumcheon-gu",
    "노원구": "Nowon-gu",
    "도봉구": "Dobong-gu",
    "동대문구": "Dongdaemun-gu",
    "동작구": "Dongjak-gu",
    "마포구": "Mapo-gu",
    "서대문구": "Seodaemun-gu",
    "서초구": "Seocho-gu",
    "성동구": "Seongdong-gu",
    "성북구": "Seongbuk-gu",
    "송파구": "Songpa-gu",
    "양천구": "Yangcheon-gu",
    "영등포구": "Yeongdeungpo-gu",
    "용산구": "Yongsan-gu",
    "은평구": "Bulgwang-dong, Eunpyeong-gu",
    "종로구": "Jongno-gu",
    "중구": "Jung-gu",
    "중랑구": "Jungnang-gu"
}

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
        """단일 입력 데이터를 받아 미세먼지 농도를 예측합니다. (pm25는 외부 API에서 받아옴)"""
        if not self.model:
            raise RuntimeError("모델이 정상적으로 로드되지 않았습니다. 서버 시작 로그를 확인해주세요.")

        # gu(한글 구명)를 영문명으로 변환
        kor_gu = input_data.구이름
        eng_gu = KOR_TO_ENG_GU.get(kor_gu)
        if not eng_gu:
            raise ValueError(f"'{kor_gu}'는 지원하지 않는 구 이름입니다. 한글로 정확히 입력해 주세요.")

        # 영문 구명과 date로 pm25 값을 외부 API에서 받아옴
        pm25_value = self.get_pm25(eng_gu, input_data.date)

        # 입력 데이터 dict로 변환 후 pm25 추가
        input_dict = input_data.dict()
        input_dict['pm25'] = pm25_value
        # gu, date는 모델 입력에 필요 없으므로 제거
        input_dict.pop('구이름', None)
        input_dict.pop('date', None)

        # DataFrame 생성
        input_df = pd.DataFrame([input_dict])
        predict_df = input_df.rename(columns={
            "기온": "평균기온(°C)",
            "강수량": "일강수량(mm)",
            "풍속": "평균 풍속(m/s)"
        })
        X = predict_df[self.feature_columns]
        prediction = self.model.predict(X)
        predicted_value = prediction[0]
        result = PredictionOutput(
            date=input_data.date,
            pm10=round(predicted_value, 2)
        )
        return result

    def predict_bundle(self, forecast_input: BundleInput) -> BundleOutput:
        """
        여러 날짜별로 구, 날짜, 기상정보를 받아 date/pm10만 리스트로 반환
        """
        results = []
        for item in forecast_input.data:
            # PredictionInput으로 변환 (pm25는 predict에서 처리)
            pred_input = PredictionInput(
                date=item.date,
                month=item.month,    
                기온=item.기온,
                강수량=item.강수량,
                풍속=item.풍속,
                구이름=item.구이름,
            )
            pred_output = self.predict(pred_input)
            results.append(PredictionOutput(date=item.date, pm10=pred_output.pm10))
        return BundleOutput(data=results)

    def get_pm25(self, gu: str, target_date: str) -> float:
        """
        주어진 구와 날짜에 대해 초미세먼지(PM2.5) 평균값을 반환합니다.
        환경변수 AQI_API_KEY에서 API 토큰을 불러옵니다.
        """
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
        api_key = os.getenv('AQI_API_KEY')
        if not api_key:
            raise ValueError('API 키가 설정되어 있지 않습니다. .env 파일을 확인하세요.')

        url = f"http://api.waqi.info/feed/{gu}, Seoul, South Korea/?token={api_key}"
        resp = requests.get(url)
        data = resp.json()

        if 'data' not in data or 'forecast' not in data['data'] or 'daily' not in data['data']['forecast'] or 'pm25' not in data['data']['forecast']['daily']:
            raise ValueError('API 응답에 PM2.5 예보 데이터가 없습니다.')

        pm25_forecasts = data['data']['forecast']['daily']['pm25']
        for item in pm25_forecasts:
            if item['day'] == target_date:
                return item['avg']
        raise ValueError(f"{target_date}에 대한 PM2.5 예보가 없습니다.")


# 앱 전체에서 사용할 모델 서비스 인스턴스를 생성합니다.
# FastAPI 앱이 시작될 때 이 코드가 실행되면서 모델이 메모리에 로드됩니다.
model_service = ModelService()