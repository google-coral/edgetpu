echo off
setlocal enabledelayedexpansion

if not defined PYTHON ( set PYTHON=python )

for /f %%i in ('bazel info output_path') do set "BAZEL_OUTPUT_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(str(sys.version_info.major)+str(sys.version_info.minor))"') do set "PY3_VER=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.executable)"') do set "PYTHON_BIN_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.base_prefix)"') do set "PYTHON_LIB_PATH=%%i\Lib"

set BAZEL_OUTPUT_PATH=%BAZEL_OUTPUT_PATH:/=\%
set CPU=x64_windows
set COMPILATION_MODE=opt
set LIBEDGETPU_VERSION=direct

set ROOTDIR=%~dp0\..\..\
set BAZEL_OUT_DIR=%BAZEL_OUTPUT_PATH%\%CPU%-%COMPILATION_MODE%\bin
set SWIG_OUT_DIR=%ROOTDIR%\edgetpu\swig
set TOOLS_OUT_DIR=%ROOTDIR%\out\%CPU%\tools
set EXAMPLES_OUT_DIR=%ROOTDIR%\out\%CPU%\examples
set TESTS_OUT_DIR=%ROOTDIR%\out\%CPU%\tests
set BENCHMARKS_OUT_DIR=%ROOTDIR%\out\%CPU%\benchmarks
set LIBEDGETPU_DIR=%ROOTDIR%\libedgetpu\%LIBEDGETPU_VERSION%\x64_windows

set SWIG_WRAPPER_NAME=_edgetpu_cpp_wrapper.cp%PY3_VER%-win_amd64.pyd
set LIBEDGETPU_DLL_NAME=edgetpu.dll
set GLOG_DLL_NAME=glog.dll

set SWIG_WRAPPER_PATH=%SWIG_OUT_DIR%\%SWIG_WRAPPER_NAME%
set LIBEDGETPU_DLL_PATH=%LIBEDGETPU_DIR%\%LIBEDGETPU_DLL_NAME%
set GLOG_DLL_PATH=%BAZEL_OUT_DIR%\external\glog\windows\Release\%GLOG_DLL_NAME%

:PROCESSARGS
set ARG=%1
if defined ARG (
    if "%ARG%"=="/DBG" (
        set COMPILATION_MODE=dbg
    )
    shift
    goto PROCESSARGS
)

set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC
set BAZEL_BUILD_FLAGS= ^
--compilation_mode=%COMPILATION_MODE% ^
--copt=/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ^
--linkopt=/DEFAULTLIB:%LIBEDGETPU_DLL_PATH%.if.lib ^
--copt=/std:c++17

rem Tests
for /F "tokens=* USEBACKQ" %%g in (`bazel query "kind(cc_.*test, //src/cpp/...)"`) do (set "tests=!tests! %%g")
bazel build %BAZEL_BUILD_FLAGS% %tests%
for /F %%i in ('dir /a:-d /s /b %BAZEL_OUT_DIR%\*_test.exe') do (
    set out_dir="%%~dpi"
    set out_dir=!out_dir:%BAZEL_OUT_DIR%\=!
    set out_dir=%TESTS_OUT_DIR%\!out_dir!
    if not exist !out_dir! md !out_dir!
    copy %%i !out_dir! >NUL
    copy %GLOG_DLL_PATH% !out_dir!\%GLOG_DLL_NAME% >NUL
)

rem Benchmarks
for /F "tokens=* USEBACKQ" %%g in (`bazel query "kind(cc_binary, //src/cpp/...)"`) do (echo %%g | findstr benchmark >NUL && set "benchmarks=!benchmarks! %%g")
bazel build %BAZEL_BUILD_FLAGS% %benchmarks%
for /F %%i in ('dir /a:-d /s /b %BAZEL_OUT_DIR%\*_benchmark.exe') do (
    set out_dir="%%~dpi"
    set out_dir=!out_dir:%BAZEL_OUT_DIR%\=!
    set out_dir=%BENCHMARKS_OUT_DIR%\!out_dir!
    if not exist !out_dir! md !out_dir!
    copy %%i !out_dir! >NUL
    copy %GLOG_DLL_PATH% !out_dir!\%GLOG_DLL_NAME% >NUL
)

rem Tools
bazel build %BAZEL_BUILD_FLAGS% ^
    //src/cpp/tools:join_tflite_models ^
    //src/cpp/tools:multiple_tpus_performance_analysis
if not exist %TOOLS_OUT_DIR% md %TOOLS_OUT_DIR%
copy %BAZEL_OUT_DIR%\src\cpp\tools\join_tflite_models.exe %TOOLS_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\src\cpp\tools\multiple_tpus_performance_analysis.exe %TOOLS_OUT_DIR% >NUL
copy %GLOG_DLL_PATH% %TOOLS_OUT_DIR%\%GLOG_DLL_NAME% >NUL

rem Examples
bazel build %BAZEL_BUILD_FLAGS% ^
    //src/cpp/examples:two_models_one_tpu ^
    //src/cpp/examples:two_models_two_tpus_threaded ^
    //src/cpp/examples:classify_image
if not exist %EXAMPLES_OUT_DIR% md %EXAMPLES_OUT_DIR%
copy %BAZEL_OUT_DIR%\src\cpp\examples\two_models_one_tpu.exe %EXAMPLES_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\src\cpp\examples\two_models_two_tpus_threaded.exe %EXAMPLES_OUT_DIR% >NUL
copy %BAZEL_OUT_DIR%\src\cpp\examples\classify_image.exe %EXAMPLES_OUT_DIR% >NUL
copy %GLOG_DLL_PATH% %EXAMPLES_OUT_DIR% >NUL

rem SWIG
bazel build %BAZEL_BUILD_FLAGS% ^
    //src/cpp/swig:all
if not exist %SWIG_OUT_DIR% md %SWIG_OUT_DIR%
copy %BAZEL_OUT_DIR%\src\cpp\swig\_edgetpu_cpp_wrapper.pyd %SWIG_WRAPPER_PATH% >NUL
copy %BAZEL_OUT_DIR%\src\cpp\swig\edgetpu_cpp_wrapper.py %SWIG_OUT_DIR%\edgetpu_cpp_wrapper.py >NUL
copy %GLOG_DLL_PATH% %SWIG_OUT_DIR% >NUL
