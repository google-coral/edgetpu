echo off
set out_dir=%~dp1\..
set build_dir=%~dp1\..\..\cmake-build
set src_dir=%~dp0\..\..\com_google_glog
mkdir %build_dir%
cd %build_dir%
cmake -DBUILD_SHARED_LIBS=1 -A x64 %src_dir%
cmake --build . --config Release
copy %build_dir%\Release\glog.dll %out_dir%\Release
copy %build_dir%\Release\glog.lib %out_dir%\Release
copy %src_dir%\src\windows\glog\*.h %out_dir%\glog
