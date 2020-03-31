echo off
setlocal enabledelayedexpansion

if not defined PY3_VER set PY3_VER=38
set ROOTDIR=%~dp0\..\..\
for /f %%i in ("%ROOTDIR%") do set "ROOTDIR=%%~fi"
for /f "tokens=2 delims==" %%i in ('wmic os get /format:value ^| findstr TotalVisibleMemorySize') do set /A "MEM_KB=%%i >> 1"

docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm -v %ROOTDIR%:c:\edgetpu -w c:\edgetpu -e PYTHON=c:\python%PY3_VER%\python.exe edgetpu-win scripts\windows\build.bat
