echo off
setlocal enabledelayedexpansion

set ROOTDIR=%~dp0\..\..\
set FAILED_TESTS=
set RETURN=0
set PYTHONPATH=%ROOTDIR%
if not defined PYTHON ( set PYTHON=python )

for /F %%i in ('dir /a:-d /s /b %ROOTDIR%\tests\*_test.py') do (
    %PYTHON% -m unittest -v tests.%%~ni
    if ERRORLEVEL 1 (
        set FAILED_TESTS=%%i;!FAILED_TESTS!
        set RETURN=1
    )
)
if %RETURN% == 1 (
    echo Failed tests: %FAILED_TESTS%
)
exit /b %RETURN%
