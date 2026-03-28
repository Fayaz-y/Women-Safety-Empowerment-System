@echo off
title Women Safety AI - Control Panel
color 0B

echo ===================================================
echo   Women Safety AI - Master System Launcher
echo ===================================================

:: Ensure we are executing in the script's directory
cd /d "%~dp0"

echo [*] Creating logs directory...
if not exist logs mkdir logs

echo [*] Cleaning up previously stuck system processes (Python/Node)...
taskkill /F /IM python.exe /T > nul 2>&1
taskkill /F /IM node.exe /T > nul 2>&1

echo [*] Starting Redis Server in WSL...
wsl -u root service redis-server start > logs\redis.log 2>&1

echo [*] Starting FastAPI Backend...
set PYTHONPATH=.
start /B cmd /c "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > logs\api.log 2>&1"

ping 127.0.0.1 -n 3 > nul

echo [*] Starting Alerts Background Worker...
set PYTHONPATH=.
start /B cmd /c "python -m alerts.dispatcher > logs\worker.log 2>&1"

echo [*] Starting Next.js Dashboard...
start /B cmd /c "cd dashboard && npm run dev > ..\logs\dashboard.log 2>&1"

echo [*] Starting AI Pipeline Engine (Main Inference Camera)...
set PYTHONPATH=.
start /B cmd /c "python scripts\run_multi_camera.py > logs\engine.log 2>&1"

echo.
echo ===================================================
echo   SUCCESS: All services started in the background!
echo ===================================================
echo.
echo   You can view everything by opening the log files:
echo    - logs\api.log       (FastAPI Backend)
echo    - logs\worker.log    (Alert worker)
echo    - logs\dashboard.log (Next.js server)
echo    - logs\engine.log    (Live tracking ^& FPS)
echo.
echo   Web UI: http://localhost:3000 
echo   (Check dashboard.log if port 3000 was occupied)
echo   Login: admin / changeme
echo.
echo ===================================================
echo   CAUTION: KEEP THIS WINDOW OPEN!
echo   When you are done, press any key below to safely 
echo   terminate all AI and Dashboard processes.
echo ===================================================
pause

echo.
echo Stopping all running AI and Dashboard services...
taskkill /F /IM python.exe /T > nul 2>&1
taskkill /F /IM node.exe /T > nul 2>&1
echo Done. Safe to close this window.
pause > nul
