# build_exe.py
import PyInstaller.__main__

PyInstaller.__main__.run([
    '--name=WeatherPredictor',
    '--onefile',
    '--add-data=app;app',
    '--add-data=fonts;fonts',
    '--add-data=app/seongnam_observed_weather.csv;app',
    '--hidden-import=sklearn.utils._typedefs',
    'app/lstm_train_weather_streamlit.py',
])
