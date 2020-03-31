echo off
setlocal enabledelayedexpansion

if not defined PYTHON ( set PYTHON=python )
set ROOTDIR=%~dp0\..\..\

rem Build the code, in case it doesn't exist yet.
call %ROOTDIR%\scripts\windows\build.bat

%PYTHON% %ROOTDIR%\setup.py bdist --bdist-base=%ROOTDIR%\build -d dist bdist_wheel
