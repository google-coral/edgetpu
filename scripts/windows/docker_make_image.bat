echo off
setlocal enabledelayedexpansion

set ROOTDIR=%~dp0\..\..\

docker build -t edgetpu-win -f %ROOTDIR%\docker\Dockerfile.windows %ROOTDIR%\docker
