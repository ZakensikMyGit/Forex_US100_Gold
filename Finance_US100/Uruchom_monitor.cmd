@echo off
setlocal
pushd "%~dp0" || exit /b 1
"%~dp0Python\python.exe" -B -m monitor %*
set "APP_STATUS=%ERRORLEVEL%"
popd
if not "%APP_STATUS%"=="0" pause
exit /b %APP_STATUS%
