@echo off
REM ===================================================================
REM Women Safety AI — Model Server + ngrok Tunnel (EASIEST WAY!)
REM ===================================================================
REM This is the simplest way to expose your model server
REM No Cloudflare setup needed!

SETLOCAL ENABLEDELAYEDEXPANSION

ECHO.
ECHO ===================================================================
ECHO  Women Safety AI — Model Server with ngrok
ECHO ===================================================================
ECHO.
ECHO This will:
ECHO   1. Start ngrok tunnel on port 9000
ECHO   2. Start Model Server
ECHO   3. Give you a public URL to share with your friend
ECHO.

REM Check if ngrok is installed
where ngrok >nul 2>nul
if errorlevel 1 (
    ECHO ERROR: ngrok not found!
    ECHO.
    ECHO Install with one of these:
    ECHO   - choco install ngrok
    ECHO   - Download from: https://ngrok.com/download
    ECHO.
    ECHO After installing, run this script again!
    ECHO.
    pause
    exit /b 1
)

ECHO [1/2] Starting ngrok tunnel on port 9000...
ECHO.
ECHO INSTRUCTIONS:
ECHO   1. Look for: "Forwarding https://XXXXXX.ngrok.io"
ECHO   2. Copy that URL
ECHO   3. Send to your friend
ECHO   4. They add it to their .env as MODEL_API_URL
ECHO.
ECHO Keep this window open!
ECHO Press Ctrl+C here ONLY to stop ngrok.
ECHO.

start cmd /k "ngrok http 9000"
timeout /t 3 >nul

ECHO.
ECHO [2/2] Starting Model Server on port 9000...
ECHO.
ECHO Wait for: "[Model Server] ✓ All models loaded successfully!"
ECHO.
ECHO Press Ctrl+C to stop.
ECHO.

python model_server.py

if errorlevel 1 (
    ECHO.
    ECHO ERROR: Model server failed!
    ECHO Check:
    ECHO   - Python installed
    ECHO   - requirements.txt installed (pip install -r requirements.txt)
    ECHO   - ./models/ folder exists
    ECHO.
    pause
    exit /b 1
)

pause
