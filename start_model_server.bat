@echo off
REM ===================================================================
REM Women Safety AI — Model Inference Server + Cloudflare Tunnel
REM ===================================================================
REM Run this on YOUR laptop to start the model server with Cloudflare
REM tunneling. The backend (on friend's laptop) will call this server.

SETLOCAL ENABLEDELAYEDEXPANSION

ECHO.
ECHO ===================================================================
ECHO  Women Safety AI — Model Inference Server (with Cloudflare Tunnel)
ECHO ===================================================================
ECHO.
ECHO This server will:
ECHO   1. Load all ML models (YOLO, Gender, Assault, Pose)
ECHO   2. Expose inference endpoints
ECHO   3. Be accessible at: https://model.yourdomain.com
ECHO.

REM Check if cloudflared is installed
where cloudflared >nul 2>nul
if errorlevel 1 (
    ECHO ERROR: cloudflared not found!
    ECHO Please install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    ECHO Or: choco install cloudflared
    ECHO.
    pause
    exit /b 1
)

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    ECHO ERROR: Python not found in PATH!
    ECHO Install Python or add it to PATH
    ECHO.
    pause
    exit /b 1
)

ECHO [1/3] Verifying Cloudflare tunnel "women-safety-model"...
cloudflared tunnel list | findstr "women-safety-model" >nul
if errorlevel 1 (
    ECHO WARNING: Tunnel "women-safety-model" not found!
    ECHO Create it with: cloudflared tunnel create women-safety-model
    ECHO Then update config.yml with ingress rules.
    ECHO.
    pause
    exit /b 1
)
ECHO [✓] Tunnel found

ECHO.
ECHO [2/3] Starting Cloudflare Tunnel...
ECHO Keep this window running in the background!
ECHO Press Ctrl+C in the new window to stop.
ECHO.
start cmd /k "cloudflared tunnel run women-safety-model"
timeout /t 2 >nul

ECHO [3/3] Starting Model Inference Server on port 9000...
ECHO Workers: 1
ECHO Device: cuda (if available) or cpu
ECHO.
ECHO Press Ctrl+C to stop.
ECHO.

python model_server.py
if errorlevel 1 (
    ECHO.
    ECHO ERROR: Model server failed to start!
    ECHO Check that all model files are in ./models/
    ECHO.
    pause
    exit /b 1
)

pause
