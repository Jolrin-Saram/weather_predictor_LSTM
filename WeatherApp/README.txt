# 🌡️ 성남시 LSTM 기온 예측 시스템

본 프로그램은 성남시 분당 지역의 과거 기온 데이터를 기반으로 LSTM 모델을 학습하고, 현재 시각 기준으로 1시간 후 기온을 예측합니다. 또한 OpenWeatherMap API를 통해 실시간 기온과 예측값을 비교합니다.

## 기능
- 과거 기온 CSV 기반 LSTM 학습
- 실시간 예보 API를 활용한 예측 비교
- 웹 UI (Streamlit 기반)
- 모델 성능 자동 저장 및 개선

## 사용 방법
1. `.exe` 파일을 실행하세요.
2. 자동으로 웹 브라우저가 열리고 Streamlit UI가 실행됩니다.
3. 기온 예측 및 실시간 기온 비교 결과를 확인하세요.

## 필요 파일
- `seongnam_observed_weather.csv`: 필수 관측 데이터
- `fonts/`: Nanum 폰트 폴더

weather_predictor/
├── app/
│   ├── lstm_train_weather_streamlit.py    ← 현재 Canvas 코드
│   ├── seongnam_observed_weather.csv      ← 관측 데이터
│   └── result/                            ← 예측 모델 저장용 (빈 폴더 가능)
├── fonts/
│   ├── NanumGothic-Regular.ttf
│   ├── NanumGothic-Bold.ttf
│   └── NanumGothic-ExtraBold.ttf
├── README.md                              ← 프로그램 설명
├── requirements.txt                       ← 필요 패키지 목록
├── run_app.bat                            ← 실행용 배치파일
└── build_exe.py                           ← exe 만들기용 스크립트