# 🖥️ VISUAL TERMINAL SETUP GUIDE

This shows exactly what you should see in each terminal.

---

## YOUR MACHINE (Running Model Server)

```
┌─────────────────────────────────────────────────────────────┐
│ Terminal 1️⃣ — CLOUDFLARE TUNNEL                      [■ □ ✕] │
├─────────────────────────────────────────────────────────────┤
│ C:\Users\You\.cloudflared>                                  │
│ cloudflared tunnel run women-safety-model                   │
│                                                              │
│ 2026-04-02T10:30:00Z INF Reading tunnel credentials       │
│ 2026-04-02T10:30:01Z INF Registering tunnel with Cloudflare│
│ 2026-04-02T10:30:02Z INF Edge assigned address ws://       │
│ 2026-04-02T10:30:03Z CONNECT INF tunnel running at         │
│                     https://model.yourdomain.com            │
│                                                              │
│ ✅ KEEP THIS RUNNING — DO NOT CLOSE                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ Terminal 2️⃣ — MODEL SERVER                           [■ □ ✕] │
├─────────────────────────────────────────────────────────────┤
│ D:\@Women_safety\women_safety>                              │
│ python model_server.py                                      │
│                                                              │
│ =====================================================         │
│  Women Safety AI — Model Inference Server                  │
│ =====================================================         │
│ Starting on http://0.0.0.0:9000                            │
│ =====================================================         │
│                                                              │
│ [Model Server] Loading models...                           │
│ [Model Server] Using device: cuda                          │
│ [Model Server] Loading YOLO detection model...             │
│ [Model Server] ✓ Loading YOLO detection model [==...]      │
│ [Model Server] Loading assault detection model...          │
│ [Model Server] ✓ Loading assault detection model [====...]  │
│ [Model Server] Loading gender classifier...                │
│ [Model Server] ✓ Loading gender classifier [======...]      │
│ [Model Server] Loading pose estimator...                   │
│ [Model Server] ✓ Loading pose estimator [=====...]          │
│ [Model Server] Initializing proximity engine...            │
│ [Model Server] ✓ All models loaded successfully!           │
│                                                              │
│ INFO:     Application startup complete                     │
│ INFO:     Uvicorn running on http://0.0.0.0:9000          │
│                                                              │
│ ✅ KEEP THIS RUNNING — DO NOT CLOSE                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## FRIEND'S MACHINE (Backend + Frontend)

```
┌─────────────────────────────────────────────────────────────┐
│ Terminal 1️⃣ — BACKEND (FastAPI)                      [■ □ ✕] │
├─────────────────────────────────────────────────────────────┤
│ C:\women_safety>                                            │
│ python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 │
│ --workers=1                                                 │
│                                                              │
│ INFO:     Started server process [12345]                   │
│ INFO:     Waiting for application startup.                 │
│ [Camera] Mode: device index 0                              │
│ [Camera] Opening camera at index 0...                      │
│ --- Starting engine for Camera 1 (source=0) ---            │
│ Loading YOLO detection model...                            │
│ [models/...] loaded successfully                           │
│ ... (more model loading)                                   │
│ [API] All camera engines started and registered.           │
│ INFO:     Application startup complete                     │
│ INFO:     Uvicorn running on http://0.0.0.0:8000          │
│          (Press CTRL+C to quit)                            │
│                                                              │
│ ✅ KEEP THIS RUNNING — DO NOT CLOSE                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ Terminal 2️⃣ — FRONTEND (Next.js)                     [■ □ ✕] │
├─────────────────────────────────────────────────────────────┤
│ C:\women_safety\dashboard>                                  │
│ npm run dev                                                 │
│                                                              │
│ > dashboard@0.1.0 dev                                       │
│ > next dev                                                  │
│                                                              │
│   ▲ Next.js 14.2.35                                        │
│   - Local:        http://localhost:3000                    │
│   - Environments: .env.local                               │
│                                                              │
│   ✓ Ready in 2.3s                                          │
│   ✓ Compiled client and server successfully                │
│                                                              │
│ ✅ KEEP THIS RUNNING — DO NOT CLOSE                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Browser (Friend's Laptop)

```
┌─────────────────────────────────────────────────────────────┐
│ http://localhost:3000                              [🔒 ⟲ ✕] │
├─────────────────────────────────────────────────────────────┤
│ Women Safety AI Dashboard                                   │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Email: [___________________________]                    │ │
│ │  Password: [________________________]                    │ │
│ │                                                          │ │
│ │  [Login Button]                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ After login, you'll see:                                   │
│ - Live camera feed                                         │
│ - Detection boxes on video                                 │
│ - Real-time alerts                                         │
│ - System status                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## WHAT HAPPENS WHEN YOU START THEM IN ORDER

### 1. Start Model Server (Your Laptop) — 2 minutes

```
⏱️ 0s:   You run: cloudflared tunnel run women-safety-model
✅ 3s:   Tunnel connects to Cloudflare
⏱️ 3s:   You run: python model_server.py
⏱️ 5s:   Starting to load YOLO model...
⏱️ 45s:  YOLO loaded
⏱️ 50s:  Loading assault model...
⏱️ 70s:  Assault model loaded
⏱️ 75s:  Loading gender classifier...
⏱️ 90s:  Gender loaded
⏱️ 95s:  Loading pose estimator...
⏱️ 110s: Pose loaded
⏱️ 115s: All models ready!
✅ 120s: Model server ready at https://model.yourdomain.com
```

### 2. Start Backend (Friend's Laptop) — 15 seconds

```
⏱️ 0s:   Friend runs: python -m uvicorn api.main:app...
⏱️ 2s:   Database connected
⏱️ 5s:   Camera opened (device 0)
⏱️ 10s:  Backend ready at http://localhost:8000
✅ 15s:  Backend waiting for frontend
```

### 3. Start Frontend (Friend's Laptop) — 10 seconds

```
⏱️ 0s:   Friend runs: npm run dev
⏱️ 3s:   Next.js compiling...
⏱️ 8s:   Compilation complete
✅ 10s:  Frontend ready at http://localhost:3000
```

### 4. Open Dashboard — Instant

```
Friend opens: http://localhost:3000
✅ Login page appears
✅ No CORS errors
✅ Camera permission dialog
✅ After login: Live feed with detections
```

---

## EXPECTED OUTPUTS (Success Markers)

### ✅ YOU SEE THIS ON YOUR MACHINE

**Cloudflare Terminal:**
```
✅ "tunnel running at https://model.yourdomain.com"
```

**Model Server Terminal:**
```
✅ "[Model Server] ✓ All models loaded successfully!"
✅ "INFO:     Uvicorn running on http://0.0.0.0:9000"
```

---

### ✅ FRIEND SEES THIS ON THEIR MACHINE

**Backend Terminal:**
```
✅ "Uvicorn running on http://0.0.0.0:8000"
✅ "[API] All camera engines started and registered"
✅ "Application startup complete"
```

**Frontend Terminal:**
```
✅ "▲ Next.js 14.2.35"
✅ "✓ Ready in 2.3s"
✅ "Local: http://localhost:3000"
```

**Browser:**
```
✅ Page loads at localhost:3000
✅ Login form appears
✅ No red errors in console (F12)
```

---

## ❌ PROBLEMS YOU MIGHT SEE

### Model Server

```
❌ "Port 9000 already in use"
   → Kill process using port 9000
   
❌ "CUDA out of memory"
   → Change DEVICE=cpu in .env, or close other GPU apps
   
❌ "No module named torch"
   → pip install -r requirements.txt
   
❌ "Models not found"
   → Check ./models/ folder has all model files
```

### Backend

```
❌ "Connection refused" (when calling model server)
   → Your model server not running or wrong URL
   → Check MODEL_API_URL in .env
   
❌ "Database connection failed"
   → PostgreSQL not running
   → Check DATABASE_URL in .env
   
❌ "Port 8000 already in use"
   → Kill process using 8000
   → lsof -i :8000 (Mac/Linux)
   → netstat -ano | findstr :8000 (Windows)
```

### Frontend

```
❌ "CORS error in console"
   → Backend ALLOWED_ORIGINS wrong
   → Restart backend after fixing .env
   
❌ "Cannot find npm"
   → Install Node.js from nodejs.org
   
❌ "Module not found"
   → npm install in dashboard folder
```

---

## 📊 RESOURCE MONITOR

While running, you should see approximately:

| Component | CPU | RAM | GPU |
|-----------|-----|-----|-----|
| Cloudflare | <5% | 20MB | — |
| Model Server | 20-40% | 4-6GB | 80-95% |
| Backend | 5-10% | 300MB | — |
| Frontend | <5% idle | 50MB | — |
| **Total** | 30-55% | 4.5-7GB | 80-95% |

Your laptop fans will be loud (that's normal — GPU working hard).

---

## 🎯 FINAL CHECKLIST BEFORE USING

Before asking your friend to use the dashboard:

- [ ] Terminal 1 (your machine): Shows "tunnel running at..."
- [ ] Terminal 2 (your machine): Shows "✓ All models loaded"
- [ ] Terminal 1 (friend's machine): Shows "Uvicorn running on 0.0.0.0:8000"
- [ ] Terminal 2 (friend's machine): Shows "Local: http://localhost:3000"
- [ ] Browser loads http://localhost:3000 without errors
- [ ] Browser console (F12) has no red errors
- [ ] Can click on login form
- [ ] Camera permission dialog appears

✅ **All 8 checks pass? System is ready!**

---

## ONE-MINUTE SURVIVAL GUIDE

If something breaks:

```
1. Ctrl+C all terminals
2. Wait 5 seconds
3. Start again in order:
   - Your machine: terminal 1 & 2
   - Friend's machine: terminal 1 & 2
4. Wait 2 minutes for models to load
5. Open http://localhost:3000
6. Should work!
```

---

## QUICK REFERENCE: TERMINAL COLORS

Use different colors for clarity:

```
Terminal Color    Purpose              Keep Open?
─────────────────────────────────────────────────
🟦 Blue         Cloudflare Tunnel    ✅ Yes
🟪 Purple       Model Server         ✅ Yes
🟥 Red          Backend              ✅ Yes
🟩 Green        Frontend             ✅ Yes
```

---

**You're ready! Follow this visual guide and everything will work!** 🚀

