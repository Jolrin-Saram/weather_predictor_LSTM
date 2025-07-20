@echo off
cd /d %~dp0app
streamlit run lstm_train_weather_streamlit.py
pause