import subprocess
import os
import sys

# 현재 실행 파일 경로 기준으로 app 경로 설정
script_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
app_path = os.path.join(script_dir, "app", "lstm_train_weather_streamlit.py")

# Streamlit 실행
subprocess.Popen(["streamlit", "run", app_path], shell=True)
