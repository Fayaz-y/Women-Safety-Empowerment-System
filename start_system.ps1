# Women Safety AI - Control Panel (PowerShell Version)
# ===================================================

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Women Safety AI - Master System Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we are in the script's directory
Set-Location -Path $PSScriptRoot

Write-Host "[*] Creating logs directory..." -ForegroundColor Yellow
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "[*] Cleaning up previously stuck system processes (Python/Node)..." -ForegroundColor Yellow
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Stop-Process -Name node -Force -ErrorAction SilentlyContinue

Write-Host "[*] Starting Redis Server in WSL..." -ForegroundColor Yellow
wsl -u root service redis-server start > "logs\redis.log" 2>&1

$env:PYTHONPATH="."

Write-Host "[*] Starting FastAPI Backend..." -ForegroundColor Yellow
Start-Process powershell -NoNewWindow -ArgumentList "-Command `"uvicorn api.main:app --host 0.0.0.0 --port 8000 > 'logs\api.log' 2>&1`""

Start-Sleep -Seconds 3

Write-Host "[*] Starting Alerts Background Worker..." -ForegroundColor Yellow
Start-Process powershell -NoNewWindow -ArgumentList "-Command `"python -m alerts.dispatcher > 'logs\worker.log' 2>&1`""

Write-Host "[*] Starting Next.js Dashboard..." -ForegroundColor Yellow
Start-Process powershell -NoNewWindow -ArgumentList "-Command `"cd dashboard; npm run dev > '..\logs\dashboard.log' 2>&1`""

Write-Host "[*] Starting AI Pipeline Engine (Main Inference Camera)..." -ForegroundColor Yellow
Start-Process powershell -NoNewWindow -ArgumentList "-Command `"python scripts\run_multi_camera.py > 'logs\engine.log' 2>&1`""

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All services started in the background!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  You can view everything by opening the log files:"
Write-Host "   - logs\api.log       (FastAPI Backend)"
Write-Host "   - logs\worker.log    (Alert worker)"
Write-Host "   - logs\dashboard.log (Next.js server)"
Write-Host "   - logs\engine.log    (Live tracking & FPS)"
Write-Host ""
Write-Host "  Web UI: http://localhost:3000"
Write-Host "  (Check dashboard.log if port 3000 was occupied)"
Write-Host "  Login: admin / changeme"
Write-Host ""
Write-Host "===================================================" -ForegroundColor Red
Write-Host "  CAUTION: KEEP THIS TERMINAL OPEN!" -ForegroundColor Red
Write-Host "  When you are done, press any key below to safely "
Write-Host "  terminate all AI and Dashboard processes."
Write-Host "===================================================" -ForegroundColor Red
Write-Host ""

# Wait for user input
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null

Write-Host "`nStopping all running AI and Dashboard services..." -ForegroundColor Yellow
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Stop-Process -Name node -Force -ErrorAction SilentlyContinue
Write-Host "Done. Fast shutdown complete." -ForegroundColor Green
