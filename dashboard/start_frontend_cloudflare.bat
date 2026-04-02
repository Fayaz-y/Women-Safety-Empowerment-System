@echo off
REM ===================================================================
REM Women Safety AI — Frontend Setup & Startup Script
REM ===================================================================
REM Use this on your friend's computer to:
REM 1. Configure the backend URL
REM 2. Install dependencies
REM 3. Start the Next.js dashboard

SETLOCAL ENABLEDELAYEDEXPANSION

ECHO.
ECHO ===================================================================
ECHO  Women Safety AI Frontend - Cloudflare Setup
ECHO ===================================================================
ECHO.

cd dashboard

REM Ask for the Cloudflare domain
SET /p DOMAIN="Enter your backend Cloudflare domain (e.g., women-safety.example.com): "

if "%DOMAIN%"=="" (
    ECHO ERROR: Domain cannot be empty!
    pause
    exit /b 1
)

REM Create .env.local
ECHO Creating .env.local...
(
    ECHO # Backend API configuration
    ECHO NEXT_PUBLIC_API_URL=https://%DOMAIN%
    ECHO NEXT_PUBLIC_WS_URL=wss://%DOMAIN%
) > .env.local

ECHO.
ECHO Created .env.local with:
type .env.local
ECHO.

REM Check if node_modules exists
if not exist "node_modules" (
    ECHO Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        ECHO ERROR: npm install failed!
        pause
        exit /b 1
    )
) else (
    ECHO Node modules already exist, skipping install
)

ECHO.
ECHO ===================================================================
ECHO  Starting Next.js Dashboard...
ECHO ===================================================================
ECHO.
ECHO Frontend will be at: http://localhost:3000
ECHO Backend is at: https://%DOMAIN%
ECHO.
ECHO Press Ctrl+C to stop the development server
ECHO.

call npm run dev

pause
