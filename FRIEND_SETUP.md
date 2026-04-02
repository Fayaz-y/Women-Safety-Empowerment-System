# Setup Guide for Your Friend (Backend + Frontend on Their Laptop)

This guide is for your friend who will be running the backend and accessing the dashboard.

---

## Overview

```
Your Laptop (With GPU)
  ↓
Model Server
  ↓ (HTTPS via Cloudflare)
Friend's Laptop
  ├── Backend (FastAPI) - port 8000
  └── Frontend (Dashboard) - port 3000
```

Your friend does NOT need a GPU. The models run on your laptop.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL & Redis (local or remote)
- Internet connection (to reach your model server)

---

## Step 1: Get the Code

Friend should clone or copy the project:
```bash
# Either git clone or copy the folder to their machine
cd women_safety
```

---

## Step 2: Get Your Model Server URL

Ask you (the machine with GPU) for your model server URL.

It will look like:
```
https://model.yourdomain.com
```

---

## Step 3: Configure Environment

In the project root, create or edit `.env`:

```env
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: Remote Model Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tell backend to use remote models from your friend's laptop
USE_REMOTE_MODEL=true
# Use the model URL they gave you
MODEL_API_URL=https://model.yourdomain.com
MODEL_API_TIMEOUT=30

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Database
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# If database is on the same laptop:
DATABASE_URL=postgresql://postgres:password@localhost:5432/women_safety
REDIS_URL=redis://localhost:6379/0

# Or if database is remote, use their URL:
# DATABASE_URL=postgresql://postgres:password@db.example.com:5432/women_safety
# REDIS_URL=redis://redis.example.com:6379/0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Camera (on THIS laptop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0 = built-in webcam
# 1 = external USB camera
# 2 = video file
CAMERA_TOGGLE=0
CAMERA_SOURCE_PATH=

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These are NOT used when USE_REMOTE_MODEL=true, but keep them:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE=cpu
USE_FP16=true
MODEL_DIR=./models
SNAPSHOT_DIR=./data/snapshots
CLIP_DIR=./data/clips

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detection Thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUSION_THRESHOLD=0.75
PROXIMITY_RADIUS_PX=150
ISOLATION_SECONDS=5.0
ALERT_COOLDOWN_SECONDS=30

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Twilio SMS (optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
ALERT_PHONE_NUMBER=

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JWT Auth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JWT_SECRET_KEY=change-me-in-production-minimum-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API URLs (local, since backend is on same machine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## Step 4: Install Dependencies

### Backend:
```bash
pip install -r requirements.txt
```

### Frontend:
```bash
cd dashboard
npm install
cd ..
```

---

## Step 5: Create Database (if needed)

If using local PostgreSQL:
```bash
python scripts/create_tables.py
```

---

## Step 6: Start Backend

**Terminal 1:**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
```

This window can stay open in the background.

---

## Step 7: Start Frontend

**Terminal 2 (in `dashboard/` folder):**
```bash
cd dashboard
npm run dev
```

You should see:
```
Local: http://localhost:3000
```

---

## Step 8: Access Dashboard

Open your browser:
```
http://localhost:3000
```

You'll see login page. The dashboard will communicate with the backend, which will call the model server on your friend's laptop (via HTTPS).

---

## What to Do If Something Goes Wrong

### ❌ Backend says "Model API failed"
```
Check:
1. The MODEL_API_URL in .env is correct
   (Should be exactly what your friend gave you)
2. Your internet connection works
3. Ask your friend to check if model server is running
   (They should see "tunnel running" message)
```

### ❌ Database connection error
```
Check:
1. PostgreSQL is running
   (or ask your friend if it's on a remote server)
2. DATABASE_URL in .env is correct
3. Create tables: python scripts/create_tables.py
```

### ❌ Camera not working
```
Check:
1. CAMERA_TOGGLE is correct (0 for webcam, 1 for USB)
2. Camera is not being used by another app
3. Give permission if Windows asks
```

### ❌ Frontend/Backend can't communicate
```
Check:
1. Both are running (check terminal windows)
2. No firewall blocking port 8000
3. Backend at http://localhost:8000 should work
```

---

## Architecture Diagram (What's Happening)

```
┌─────────────────────────────────┐
│  Your Laptop (YOUR FRIEND'S)    │
│                                 │
│  ┌──────────────────────────┐   │
│  │  Dashboard (port 3000)   │   │
│  │  (Browser View)          │   │
│  └────────────┬─────────────┘   │
│               │ HTTP             │
│  ┌────────────▼─────────────┐   │
│  │  Backend (port 8000)     │   │
│  │  - Captures camera       │   │
│  │  - Processes data        │   │
│  │  - Stores alerts         │   │
│  └────────────┬─────────────┘   │
│               │ HTTPS (encrypted)
└───────────────┼─────────────────┘
                │
         (INTERNET)
                │
┌───────────────▼─────────────────┐
│  YOUR Laptop (GPU)              │
│                                 │
│  ┌──────────────────────────┐   │
│  │  Model Server            │   │
│  │  (port 9000)             │   │
│  │                          │   │
│  │  ✓ YOLO Detection        │   │
│  │  ✓ Gender Classification │   │
│  │  ✓ Assault Detection     │   │
│  │  ✓ Pose Estimation       │   │
│  └────────────┬─────────────┘   │
│               │                 │
│  ┌────────────▼─────────────┐   │
│  │  Cloudflare Tunnel       │   │
│  │  (exposes to Internet)   │   │
│  └──────────────────────────┘   │
│                                 │
│  URL: https://model.yourdomain  │
└─────────────────────────────────┘
```

---

## Next Steps

1. ✅ Confirm `.env` has `USE_REMOTE_MODEL=true`
2. ✅ Confirm `MODEL_API_URL` is correct
3. ✅ Start backend
4. ✅ Start frontend
5. ✅ Test at `http://localhost:3000`

If database setup is needed, run:
```bash
python scripts/seed_db.py  # Create default data
```

---

## Keep in Mind

- **Both terminals must stay running** (backend + frontend)
- **Internet connection required** to reach model server
- **No GPU needed** on this laptop
- **Can close dashboard**, backend stays running and keeps processing
- **Restart backend if** `.env` is changed

---

## Support

If you have issues:

1. **Check all 3 things are running:**
   - Your friend's model server (tunnel + model_server.py)
   - Your backend (terminal 1)
   - Your frontend dev server (terminal 2)

2. **Check network:**
   - Internet working?
   - Can open `https://model.yourdomain.com/` in browser?

3. **Check logs:**
   - Backend terminal for errors
   - Browser console (F12) for frontend errors
   - Model server terminal on your friend's laptop

4. **Restart in order:**
   - Stop frontend
   - Stop backend
   - Start backend
   - Start frontend

Good luck! 🚀

