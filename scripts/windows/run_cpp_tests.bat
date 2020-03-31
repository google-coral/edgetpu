echo off
setlocal enabledelayedexpansion

set CPU=x64_windows

set ROOTDIR=%~dp0\..\..\
set TESTS_OUT_DIR=%ROOTDIR%\out\%CPU%\tests
set FAILED_TESTS=
set RETURN=0

for /F %%i in ('dir /a:-d /s /b %TESTS_OUT_DIR%\*_test.exe') do (
    %%i
    if ERRORLEVEL 1 (
        set FAILED_TESTS=%%i;!FAILED_TESTS!
        set RETURN=1
    )
)
if %RETURN% == 1 (
    echo Failed tests: %FAILED_TESTS%
)
exit /b %RETURN%