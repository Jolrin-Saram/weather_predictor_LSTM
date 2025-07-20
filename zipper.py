import os
from zipfile import ZipFile

# 디렉토리 재생성 (세션 리셋됨)
base_dir = "/mnt/data/lstm_weather_package"
os.makedirs(base_dir, exist_ok=True)

structure = {
    "result/": "",  # 모델 저장 폴더
    "docs/": "README.txt",  # 설명서 포함
    "src/": [
        "lstm_train_weather_streamlit.py",
        "main.py"
    ],
    "data/": "seongnam_observed_weather.csv"
}

# 샘플 설명서
readme_text = """📘 LSTM 기온 예측 시스템

이 프로그램은 성남시 분당 지역의 과거 기온 데이터를 기반으로 LSTM 딥러닝 모델을 학습하고,
최신 데이터를 통해 실시간 기온을 예측합니다.

📌 주요 기능
- 과거 기온 CSV 기반 모델 학습
- 예측값과 OpenWeatherMap API의 실시간 기온 비교
- Streamlit 웹 대시보드에서 결과 시각화

📦 구성 파일
- src/lstm_train_weather_streamlit.py: 핵심 모델 + 웹 UI 코드
- src/main.py: 실행용 Python 파일
- data/seongnam_observed_weather.csv: 학습용 관측 데이터
- result/: 모델 파일 저장용 폴더

🛠 사용 방법
1. 실행:
   main.exe 또는 main.py 실행 시 Streamlit 앱이 열립니다.
2. 필요시 pip로 의존성 설치:
   pip install -r requirements.txt

© 2025 성남시 기온예측 시스템
"""

main_launcher = """import os
os.system("streamlit run lstm_train_weather_streamlit.py")
"""

lstm_streamlit_code = '''import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import os
import streamlit as st

st.set_page_config(page_title="LSTM 기온 예측", layout="centered")
st.title("🌡️ 성남시 기온 예측 시스템 (LSTM 기반)")

observed_path = "data/seongnam_observed_weather.csv"
df = pd.read_csv(observed_path, encoding='cp949')
df = df.rename(columns={"일시": "날짜", "기온(°C)": "기온"})
df["날짜"] = pd.to_datetime(df["날짜"])
df = df[["날짜", "기온"]].sort_values("날짜")

scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(df["기온"].values.reshape(-1, 1))

def make_sequences(data, window_size):
    X, y, timestamps = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        timestamps.append(df["날짜"].iloc[i+window_size])
    return np.array(X), np.array(y), timestamps

window_size = 24
X, y, timestamps = make_sequences(temp_scaled, window_size)
X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(X, y, timestamps, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

pred = model.predict(X_test)
predicted_temp = scaler.inverse_transform(pred)
actual_temp = scaler.inverse_transform(y_test)

result_df = pd.DataFrame({
    "날짜": ts_test,
    "실제기온": actual_temp.flatten(),
    "예측기온": predicted_temp.flatten()
})
st.subheader("📊 검증 결과 샘플")
st.dataframe(result_df.head(20))

if not os.path.exists("result"):
    os.makedirs("result")
model.save("result/lstm_weather_model.h5")
st.success("✅ LSTM 모델 학습 및 저장 완료")

now = datetime.now().replace(minute=0, second=0, microsecond=0)
latest_24 = df[df["날짜"] <= now].tail(window_size)
if len(latest_24) < window_size:
    st.error("❌ 최근 24시간 데이터가 부족합니다.")
    st.stop()

latest_input = scaler.transform(latest_24["기온"].values.reshape(-1, 1)).reshape(1, window_size, 1)
next_temp_scaled = model.predict(latest_input)
next_temp = scaler.inverse_transform(next_temp_scaled)[0][0]

st.subheader("📅 예측 결과")
st.metric(label="예측 기온 (다음 시각)", value=f"{next_temp:.2f} °C")

api_key = "b7973cb99f7ff11c8c1d685442058653"
lat, lon = 37.3781, 127.1125
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

try:
    response = requests.get(weather_url)
    response.raise_for_status()
    data = response.json()
    realtime_temp = data['main']['temp']
    weather_desc = data['weather'][0]['description']
    delta = abs(realtime_temp - next_temp)
    
    st.subheader("🌤️ 실시간 API 기온")
    st.metric(label="실시간 측정 기온", value=f"{realtime_temp:.2f} °C")
    st.write(f"상태: {weather_desc}")
    st.warning(f"예측 기온 vs 실시간 차이: {delta:.2f} °C")

except Exception as e:
    st.error(f"❌ 실시간 API 요청 실패: {e}")
'''

# 파일 작성
os.makedirs(os.path.join(base_dir, "docs"), exist_ok=True)
with open(os.path.join(base_dir, "docs/README.txt"), "w", encoding="utf-8") as f:
    f.write(readme_text)

os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)
with open(os.path.join(base_dir, "src/main.py"), "w", encoding="utf-8") as f:
    f.write(main_launcher)

with open(os.path.join(base_dir, "src/lstm_train_weather_streamlit.py"), "w", encoding="utf-8") as f:
    f.write(lstm_streamlit_code)

os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
with open(os.path.join(base_dir, "data/seongnam_observed_weather.csv"), "w", encoding="utf-8") as f:
    f.write("일시,기온(°C)\n")

os.makedirs(os.path.join(base_dir, "result"), exist_ok=True)

# 압축
zip_path = "/mnt/data/lstm_weather_package.zip"
with ZipFile(zip_path, "w") as zipf:
    for root, _, files in os.walk(base_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, base_dir)
            zipf.write(abs_path, arcname=rel_path)

zip_path
