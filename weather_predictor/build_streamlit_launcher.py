import PyInstaller.__main__

PyInstaller.__main__.run([
    'start_streamlit_launcher.py',
    '--name=WeatherUI',
    '--onefile'
])
