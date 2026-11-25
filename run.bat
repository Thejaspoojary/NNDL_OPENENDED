@echo off
cd /d "%~dp0"
title Chest X-Ray Pneumonia Detection Web App
echo ========================================================
echo Starting Chest X-Ray Pneumonia Detection Web Application
echo ========================================================
echo Web App URL: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
echo ========================================================
echo.

:: Automatically open browser after server boots
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

.\.venv\Scripts\python.exe app.py
pause
