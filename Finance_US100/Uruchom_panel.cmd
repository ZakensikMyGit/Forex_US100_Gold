@echo off
setlocal
pushd "%~dp0" || exit /b 1
"%~dp0Python\python.exe" -B -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false
set "APP_STATUS=%ERRORLEVEL%"
popd
if not "%APP_STATUS%"=="0" pause
exit /b %APP_STATUS%
