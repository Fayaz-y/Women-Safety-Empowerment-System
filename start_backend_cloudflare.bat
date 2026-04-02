@echo off
REM ===================================================================
REM Women Safety AI — Cloudflare Backend Startup Script
REM ===================================================================
REM Run this to start both the Cloudflare tunnel and FastAPI backend
REM Requires:
REM   - cloudflared installed and authenticated
REM   - Python environment with dependencies installed
REM   - .env file configured with your Cloudflare tunnel details

ECHO.
ECHO ===================================================================
ECHO  Women Safety AI Backend — Starting with Cloudflare Tunnel
ECHO ===================================================================
ECHO.

REM Check if cloudflared is installed
where cloudflared >nul 2>nul
if errorlevel 1 (
    ECHO ERROR: cloudflared not found!
    ECHO Please install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    ECHO.
    pause
    exit /b 1
)

ECHO [1/3] Cloudflare tunnel credentials...
cloudflared tunnel list
ECHO.

ECHO [2/3] Starting Cloudflare Tunnel (women-safety-backend)
ECHO Press Ctrl+C in the new window to stop.
ECHO.
start cmd /k "cloudflared tunnel run women-safety-backend"
timeout /t 2 >nul

ECHO [3/3] Starting FastAPI Backend on port 8000
ECHO Press Ctrl+C to stop.
ECHO.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1

pause
