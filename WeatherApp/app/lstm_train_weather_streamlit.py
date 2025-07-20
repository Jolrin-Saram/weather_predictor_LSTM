import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from dateutil import parser
import os
import csv
import tempfile
import streamlit as st

st.set_page_config(page_title="LSTM 기온 예측", layout="centered")
st.title("🌡️ 성남시 기온 예측 시스템 (LSTM 기반)")

# 1. 관측 데이터 불러오기
observed_path = "seongnam_observed_weather.csv"
df = pd.read_csv(observed_path, encoding='cp949')
df = df.rename(columns={"일시": "날짜", "기온(°C)": "기온"})
df["날짜"] = pd.to_datetime(df["날짜"])
df = df[["날짜", "기온"]].sort_values("날짜")

# 2. 정규화
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(df["기온"].values.reshape(-1, 1))

# 3. 시퀀스 생성 (Sliding Window)
def make_sequences(data, window_size):
    X, y, timestamps = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        timestamps.append(df["날짜"].iloc[i+window_size])
    return np.array(X), np.array(y), timestamps

window_size = 24
X, y, timestamps = make_sequences(temp_scaled, window_size)

# 4. 학습/검증 분리
X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(X, y, timestamps, test_size=0.2, shuffle=False)

# 5. 모델 정의 및 학습
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# 6. 예측 및 결과 확인
pred = model.predict(X_test)
predicted_temp = scaler.inverse_transform(pred)
actual_temp = scaler.inverse_transform(y_test)

# 7. 결과를 표로 출력
result_df = pd.DataFrame({
    "날짜": ts_test,
    "실제기온": actual_temp.flatten(),
    "예측기온": predicted_temp.flatten()
})
st.subheader("📊 검증 결과 샘플")
st.dataframe(result_df.tail(20))

# 8. 모델 저장
if not os.path.exists("result"):
    os.makedirs("result")
model.save("result/lstm_weather_model.h5")
st.success("✅ LSTM 모델 학습 및 저장 완료")

# 9. 최신 관측 데이터 기반 예측 (현재 시각 기준)
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

# 10. OpenWeatherMap API로 현재 기온 불러오기
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
