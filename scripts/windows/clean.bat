echo off
setlocal enabledelayedexpansion

set ROOTDIR=%~dp0\..\..\

bazel clean

for /f %%i in ('dir /a:d /b %ROOTDIR%\bazel-*') do rd /q %%i
del /s /q %ROOTDIR%\edgetpu\swig\*.pyd
del /s /q %ROOTDIR%\edgetpu\swig\*.dll
del /s /q %ROOTDIR%\edgetpu\swig\edgetpu_cpp_wrapper.py
rd /s /q %ROOTDIR%\out
rd /s /q %ROOTDIR%\build
rd /s /q %ROOTDIR%\dist
