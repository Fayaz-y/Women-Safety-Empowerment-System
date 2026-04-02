# 🚀 Using NGROK Instead of Cloudflare (MUCH SIMPLER!)

**ngrok is way easier than Cloudflare. This is the recommended approach for quick testing.**

---

## ✨ Why ngrok is Better for Your Setup

| Feature | Cloudflare | ngrok |
|---------|-----------|-------|
| Setup time | 20 minutes | 2 minutes |
| DNS configuration | ✅ Required | ❌ Not needed |
| Free tier | ✅ Yes | ✅ Yes |
| Tunnel reuse | ✅ Always same URL | ❌ Changes each restart |
| Complexity | 🔴 Hard | 🟢 Easy |
| Good for testing | ⚠️ Overkill | ✅ Perfect |

---

## 📦 Installation (2 minutes)

### Windows:

**Option 1: Chocolatey (easiest)**
```bash
choco install ngrok
```

**Option 2: Direct Download**
1. Go to: https://ngrok.com/download
2. Download for Windows
3. Extract to: `C:\ngrok\`
4. Add to PATH or use full path

### Mac/Linux:
```bash
# Mac
brew install ngrok/ngrok/ngrok

# Linux
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
```

### Verify Installation:
```bash
ngrok --version
# Should show: ngrok version X.X.X
```

---

## 🔑 Free Account Setup (2 minutes)

1. Go to: https://ngrok.com
2. Click **Sign Up** (top right)
3. Sign up with email
4. Verify email
5. Copy your **Auth Token** from dashboard

---

## 🔐 Add Auth Token (1 minute)

Open terminal and run:
```bash
ngrok config add-authtoken <YOUR_AUTH_TOKEN>
```

(Replace `<YOUR_AUTH_TOKEN>` with what you copied from ngrok dashboard)

---

## 🚀 Start ngrok Tunnel (5 seconds!)

That's it! Just run:

```bash
ngrok http 9000
```

You'll see:
```
ngrok                                       (Ctrl+C to quit)

Session Status                online
Account                       <your-email>
Version                        3.5.0
Region                         us (United States)
Forwarding                     https://abc123def456.ngrok.io -> http://localhost:9000
Forwarding                     http://abc123def456.ngrok.io -> http://localhost:9000

Connections                    ttl    opn    rt1    rt5    p50    p95
                                       0      0      0.00   0.00   0.00
```

✅ **Your model server is now at:** `https://abc123def456.ngrok.io`

---

## 🎯 Complete Setup (Summary)

### Step 1: Install ngrok
```bash
choco install ngrok
```

### Step 2: Sign up at ngrok.com
- Get auth token

### Step 3: Add auth token
```bash
ngrok config add-authtoken <token>
```

### Step 4: Start ngrok tunnel
```bash
ngrok http 9000
```

### Step 5: Note your URL
```
https://abc123def456.ngrok.io
```

---

## 🏃 RUNNING EVERYTHING WITH NGROK

### YOUR LAPTOP (Model Server)

**Terminal 1 (ngrok):**
```bash
ngrok http 9000
```
Wait for the URL. Example output:
```
Forwarding     https://abc123def456.ngrok.io -> http://localhost:9000
```

Copy this URL: `https://abc123def456.ngrok.io`

**Terminal 2 (Model Server):**
```bash
cd d:\@Women_safety\women_safety
python model_server.py
```

---

### FRIEND'S LAPTOP (Backend + Frontend)

**Update their `.env` with your ngrok URL:**
```env
USE_REMOTE_MODEL=true
MODEL_API_URL=https://abc123def456.ngrok.io
```

**Terminal 1 (Backend):**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

**Terminal 2 (Frontend):**
```bash
cd dashboard
npm run dev
```

---

## 🧪 Test It Works

From friend's laptop:
```bash
curl https://abc123def456.ngrok.io/
# Should return: {"service":"Women Safety AI — Model Inference Server"...}
```

---

## ⚠️ ngrok Free Plan Limitations

| Limit | Details |
|-------|---------|
| URL changes | Each time you restart (fine, just update .env) |
| Connections | 20 concurrent OK |
| Bandwidth | Unlimited |
| Session time | 2 hours max (restart to continue) |

**For your use case:** Perfectly fine! Just restart when needed.

---

## 🔄 Workflow with ngrok

Every time you restart:

1. **Terminal 1:** `ngrok http 9000` → **Get new URL**
2. **Terminal 2:** `python model_server.py` → **Start models**
3. **Send friend the new URL** via message
4. **Friend updates .env:** `MODEL_API_URL=https://<new-url>`
5. **Friend restarts backend**

Takes 30 seconds total.

---

## 📜 Create Startup Script for ngrok

**Windows: `start_ngrok_model_server.bat`**

```batch
@echo off
REM ===================================================================
REM Women Safety AI — Model Server with ngrok Tunnel (SIMPLE!)
REM ===================================================================

ECHO.
ECHO ===================================================================
ECHO  Starting ngrok tunnel on port 9000...
ECHO ===================================================================
ECHO.

REM Check if ngrok is installed
where ngrok >nul 2>nul
if errorlevel 1 (
    ECHO ERROR: ngrok not found!
    ECHO Install with: choco install ngrok
    ECHO Or download from: https://ngrok.com/download
    pause
    exit /b 1
)

ECHO Starting ngrok...
ECHO Copy the HTTPS URL from below and share with your friend!
ECHO.
start cmd /k "ngrok http 9000"
timeout /t 2

ECHO.
ECHO Starting Model Server...
ECHO.

python model_server.py

pause
```

Save as: `start_ngrok_model_server.bat`

Run with:
```bash
.\start_ngrok_model_server.bat
```

---

## 🎯 Side-by-Side Comparison

### OLD WAY (Cloudflare)
```
1. Install cloudflared
2. cloudflared tunnel login
3. cloudflared tunnel create women-safety-model
4. Edit config.yml
5. Add DNS CNAME in Cloudflare dashboard
6. Wait for DNS propagation
7. cloudflared tunnel run women-safety-model
8. Same URL forever (good for production)
```
⏱️ **20 minutes setup, complex**

---

### NEW WAY (ngrok)
```
1. choco install ngrok
2. Sign up at ngrok.com, copy token
3. ngrok config add-authtoken <token>
4. ngrok http 9000
5. Share the URL
```
⏱️ **5 minutes setup, dead simple**

---

## 💡 When to Use Which?

### Use **ngrok** if:
- ✅ Quick testing/development
- ✅ Don't need permanent URL
- ✅ Want simplicity
- ✅ Friend's laptop changes IP
- ✅ You just want to get it working

### Use **Cloudflare** if:
- ✅ Production deployment
- ✅ Need same URL always
- ✅ Want more features
- ✅ Planning long-term use
- ❌ Don't want URL changing on restart

---

## 🚀 Recommended: Start with ngrok!

Once it works with ngrok, you can upgrade to Cloudflare later if needed.

**For now: Use ngrok. It's perfect.**

---

## Full Command Reference

### Installation
```bash
choco install ngrok                              # Install
ngrok config add-authtoken <TOKEN>              # Setup
```

### Running
```bash
ngrok http 9000                                  # Start tunnel
python model_server.py                           # Start models (different terminal)
```

### Testing
```bash
curl https://<your-ngrok-url>/                 # Test connection
```

### Cleanup
```bash
Ctrl+C in ngrok terminal                        # Stop tunnel
Ctrl+C in model terminal                        # Stop models
```

---

## ⚡ 30-Second Quick Start

1. **Install:** `choco install ngrok`
2. **Sign up:** https://ngrok.com → Copy auth token
3. **Configure:** `ngrok config add-authtoken <token>`
4. **Terminal 1:** `ngrok http 9000`
5. **Copy URL** from terminal
6. **Terminal 2:** `python model_server.py`
7. **Send URL to friend** → They add to `.env`
8. **Friend runs backend** → Done!

---

## 🎬 That's Literally It!

ngrok is:
- ✅ Instant
- ✅ No configuration
- ✅ No DNS setup
- ✅ No account management
- ✅ Perfect for your use case

**Just use this instead of Cloudflare. Much simpler!**

