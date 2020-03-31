echo off
setlocal enabledelayedexpansion

set ROOTDIR=%~dp0\..\..\
for /f %%i in ("%ROOTDIR%") do set "ROOTDIR=%%~fi"
for /f "tokens=2 delims==" %%i in ('wmic os get /format:value ^| findstr TotalVisibleMemorySize') do set /A "MEM_KB=%%i >> 1"

rem Build Python 3.5 wheel
bazel clean
del %ROOTDIR%\edgetpu\swig\*.pyd
del /s /q %ROOTDIR%\build\
del /s /q %ROOTDIR%\edgetpu.egg-info
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm -v %ROOTDIR%:c:\edgetpu -w c:\edgetpu -e PYTHON=c:\python35\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.6 wheel
bazel clean
del %ROOTDIR%\edgetpu\swig\*.pyd
del /s /q %ROOTDIR%\build\
del /s /q %ROOTDIR%\edgetpu.egg-info
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm -v %ROOTDIR%:c:\edgetpu -w c:\edgetpu -e PYTHON=c:\python36\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.7 wheel
bazel clean
del %ROOTDIR%\edgetpu\swig\*.pyd
del /s /q %ROOTDIR%\build\
del /s /q %ROOTDIR%\edgetpu.egg-info
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm -v %ROOTDIR%:c:\edgetpu -w c:\edgetpu -e PYTHON=c:\python37\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.8 wheel
bazel clean
del %ROOTDIR%\edgetpu\swig\*.pyd
del /s /q %ROOTDIR%\build\
del /s /q %ROOTDIR%\edgetpu.egg-info
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm -v %ROOTDIR%:c:\edgetpu -w c:\edgetpu -e PYTHON=c:\python38\python.exe edgetpu-win scripts\windows\build_wheel.bat
