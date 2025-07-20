[Setup]
AppName=Weather LSTM Forecast
AppVersion=1.0
DefaultDirName={pf}\WeatherApp
DefaultGroupName=Weather Forecast
OutputDir=dist
OutputBaseFilename=WeatherAppInstaller
Compression=lzma
SolidCompression=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
Source: "app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs
Source: "fonts\*"; DestDir: "{app}\fonts"; Flags: ignoreversion
Source: "D:\weather\weather_predictor\app\seongnam_observed_weather.csv"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "run_app.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.txt"; DestDir: "{app}"; Flags: ignoreversion

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 생성"; GroupDescription: "추가 아이콘:"

[Icons]
Name: "{group}\WeatherApp 실행"; Filename: "{app}\run_app.bat"
Name: "{group}\설명서"; Filename: "{app}\README.txt"
Name: "{userdesktop}\WeatherApp 실행"; Filename: "{app}\run_app.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\run_app.bat"; Description: "앱 실행"; Flags: nowait postinstall skipifsilent
