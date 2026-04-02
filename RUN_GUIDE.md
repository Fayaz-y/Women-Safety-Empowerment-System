# How to Run Frontend, Backend, and Model Server

This guide shows **exactly which commands to run in which terminals**.

---

## 🎯 Overview: Who Runs What?

```
YOU (Your Laptop with GPU)          FRIEND (Their Laptop)
├── Terminal 1: Cloudflare          ├── Terminal 1: Backend
├── Terminal 2: Model Server        └── Terminal 2: Frontend
```

---

## 🚀 OPTION 1: QUICKEST START (Recommended)

### YOUR LAPTOP (Model Server)

**Option A - One Line (if you set up Cloudflare before):**
```bash
.\start_model_server.bat
```

This automatically opens 2 windows:
- Window 1: Cloudflare tunnel
- Window 2: Model server

**Option B - Manual (step by step):**

**Terminal 1:**
```bash
cloudflared tunnel run women-safety-model
```
Keep this running. Wait until you see: `tunnel running at https://model.yourdomain.com`

**Terminal 2 (new terminal in project folder):**
```bash
python model_server.py
```

Wait until you see:
```
[Model Server] ✓ All models loaded successfully!
```

**Verify it works:**
```bash
curl https://model.yourdomain.com/
```

---

### FRIEND'S LAPTOP (Backend + Frontend)

**Terminal 1 (Backend):**

First, update `.env` file with:
```env
USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
```

Then run:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

Wait until you see:
```
Uvicorn running on http://0.0.0.0:8000
```

Keep this running.

**Terminal 2 (Frontend):**
```bash
cd dashboard
npm install
npm run dev
```

Wait until you see:
```
Local: http://localhost:3000
```

Keep this running.

**Access dashboard:**
Open browser: `http://localhost:3000`

---

## 🎬 OPTION 2: DETAILED WALKTHROUGH

### STEP 1: Prepare YOUR Laptop

**Before doing anything:**
1. Make sure you have Cloudflare tunnel set up (see DISTRIBUTED_SETUP.md)
2. Model files in `./models/` directory
3. `.env` file ready

----

### STEP 2: Start Model Server (YOU)

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux):

```bash
# Navigate to project
cd d:\@Women_safety\women_safety

# Option A: Use startup script (one-click)
start_model_server.bat

# Option B: Manual (two separate commands)
# Terminal 1:
cloudflared tunnel run women-safety-model

# Terminal 2 (open new one):
python model_server.py
```

**What you should see:**

Terminal 1 (Cloudflare):
```
2026-04-02 10:30:00 CONNECT   INF tunnel running at https://model.yourdomain.com
```

Terminal 2 (Model Server):
```
[Model Server] Loading models...
[Model Server] Using device: cuda
[Model Server] Loading YOLO detection model...
[Model Server] Loading assault detection model...
[Model Server] Loading gender classifier...
[Model Server] Loading pose estimator...
[Model Server] Initializing proximity engine...
[Model Server] ✓ All models loaded successfully!

========================================================================
Women Safety AI — Model Inference Server
========================================================================
Starting on http://0.0.0.0:9000
========================================================================
```

✅ **Your model server is running!**

---

### STEP 3: Prepare FRIEND'S Machine

Send your friend:
1. The project code
2. Tell them: `Your model server is at: https://model.yourdomain.com`

They should:
1. Copy project to their machine
2. Edit `.env` file with your model URL

---

### STEP 4: Start Backend (FRIEND)

On their laptop, open **Command Prompt**:

```bash
# Navigate to project
cd path\to\women_safety

# Start backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

**What they should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started server process [1234]
INFO:     Waiting for application startup.
[Camera] Mode: device index 0
[API] All camera engines started and registered.
INFO:     Application startup complete
```

✅ **Backend is running on port 8000!**

---

### STEP 5: Start Frontend (FRIEND)

On their laptop, open **another Command Prompt**:

```bash
# Navigate to dashboard folder
cd path\to\women_safety\dashboard

# Install dependencies (first time only)
npm install

# Start frontend
npm run dev
```

**What they should see:**
```
  ▲ Next.js 14.2.35

  ▲ Local:        http://localhost:3000
  ✓ Ready in 2.3s
```

✅ **Frontend is running on port 3000!**

---

## ✨ Full System is Now Running!

### Access Points:

| Component | URL | Machine |
|-----------|-----|---------|
| Frontend | `http://localhost:3000` | Friend's laptop (browser) |
| Backend | `http://localhost:8000` | Friend's laptop (backend only) |
| Model Server | `https://model.yourdomain.com` | Your laptop (internal) |

---

## 📋 Checklist: All Running?

Go through this list:

### YOU:
- [ ] Terminal 1: Cloudflare tunnel running
  - [ ] Shows: "tunnel running at https://model.yourdomain.com"
  - [ ] No errors
  
- [ ] Terminal 2: Model server running
  - [ ] Shows: "[Model Server] ✓ All models loaded successfully!"
  - [ ] No errors

**Test:** 
```bash
curl https://model.yourdomain.com/
# Returns: {"service":"Women Safety AI — Model Inference Server"...}
```

---

### FRIEND:
- [ ] Terminal 1: Backend running
  - [ ] Shows: "Uvicorn running on http://0.0.0.0:8000"
  - [ ] Shows: "[API] All camera engines started"
  - [ ] No CORS errors
  - [ ] No database errors

- [ ] Terminal 2: Frontend running
  - [ ] Shows: "Local: http://localhost:3000"
  - [ ] No errors

**Test:**
```bash
curl http://localhost:8000/
# Returns: {"service":"Women Safety AI API"...}
```

---

## 🌐 Open Dashboard

On friend's laptop, open browser:
```
http://localhost:3000
```

You should see:
1. **Login page** loads ✓
2. **No CORS errors** in console (F12) ✓
3. **Can click inputs** ✓
4. **Camera permission requested** ✓
5. **Dashboard loads after login** ✓

---

## ⚡ Common Issues While Running

### ❌ "Port 8000 already in use"
```bash
# Find what's using it
netstat -ano | findstr :8000

# Kill it (replace PID)
taskkill /PID 1234 /F

# Try again
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### ❌ "ModuleNotFoundError"
```bash
# Install dependencies
pip install -r requirements.txt
```

### ❌ "Cannot find Python"
```bash
# Add Python to PATH or use full path:
C:\Python310\python.exe -m uvicorn api.main:app...
```

### ❌ "Cannot find npm"
```bash
# Node.js not installed
# Download from: https://nodejs.org/
# Then try again:
npm run dev
```

### ❌ "Model API failed" (in backend)
1. Check YOUR model server is running
2. Check friend's `.env` has correct `MODEL_API_URL`
3. Restart friend's backend

### ❌ "Camera not working"
1. Allow camera permission when prompted
2. Check `CAMERA_TOGGLE=0` in `.env`
3. Try closing/opening Firefox or Chrome

---

## 🔄 Daily Workflow

### Morning (Start Everything):
```
YOU:
1. Open Command Prompt → startt_model_server.bat
2. Wait for "tunnel running" message
3. Keep both windows open

FRIEND:
4. Open Command Prompt #1 → python -m uvicorn api.main:app...
5. Open Command Prompt #2 → cd dashboard && npm run dev
6. Open browser → http://localhost:3000
```

### Evening (Stop Everything):
```
YOU:
1. Ctrl+C in Terminal 1 (Cloudflare)
2. Ctrl+C in Terminal 2 (Model Server)

FRIEND:
3. Ctrl+C in Terminal 1 (Backend)
4. Ctrl+C in Terminal 2 (Frontend)
```

---

## 📊 Resource Usage While Running

| Component | CPU | RAM | GPU | Network |
|-----------|-----|-----|-----|---------|
| Model Server | Medium | 3-6 GB | Heavy | Idle until request |
| Backend | Low | 200-500 MB | None | Sending frames |
| Frontend | None | Minimal | None | Receiving data |

**Expected latency:**
- Single frame inference: 100-300ms
- Dashboard response: <100ms
- Overall feels instant to user

---

## 🧪 Testing Inference

After everything is running:

### From Friend's Dashboard:
1. Login
2. Go to "Live Monitor" 
3. Allow camera access
4. Should see camera feed
5. If people detected → boxes appear
6. Alerts should appear if threat detected

### From Your Terminal (to test model):
```bash
# Test detection endpoint
curl -X POST https://model.yourdomain.com/api/v1/health
# Should return: {"status":"ok","models":{"detection":true,...},"device":"cuda"}
```

---

## 💡 Pro Tips

1. **Keep terminals organized:**
   - Use different colors for different terminals
   - Label them: "Model Server", "Backend", "Frontend"

2. **Watch the logs:**
   - Model server: Shows inference timing
   - Backend: Shows camera frames processed
   - Frontend: Browser console shows errors

3. **Keyboard shortcuts:**
   - Ctrl+C: Stop a service
   - Ctrl+K: Clear terminal
   - F12: Open browser console (for frontend debugging)

4. **Check connectivity:**
   ```bash
   # From friend's machine:
   ping model.yourdomain.com
   curl https://model.yourdomain.com/
   ```

---

## 🐛 Debug Mode (If Something Breaks)

Add verbosity to see more details:

**Backend (Friend):**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

**Frontend (Friend):**
```bash
cd dashboard
npm run dev  # Already verbose
# Check browser console: F12 → Console tab
```

**Model Server (You):**
```bash
cloudflared tunnel run women-safety-model --loglevel debug
```

---

## ✅ Success Indicators

Everything is working when:

- ✅ Model server terminal shows: "tunnel running at https://model.yourdomain.com"
- ✅ Backend terminal shows: "Uvicorn running on http://0.0.0.0:8000"
- ✅ Frontend terminal shows: "Local: http://localhost:3000"
- ✅ Browser shows dashboard at localhost:3000
- ✅ Camera feed loads without errors
- ✅ No CORS errors in browser console
- ✅ FPS counter updates (inference running)
- ✅ Detections appear as boxes on video

**Congratulations! System is fully operational!** 🎉

---

## Next Steps After Running

1. **Test with different scenarios:**
   - Bright light
   - Dark light
   - Crowded area
   - Single person
   - Multiple people

2. **Fine-tune thresholds** if needed:
   - Edit `.env`: `FUSION_THRESHOLD=0.75`
   - Restart backend
   - Test again

3. **Monitor performance:**
   - Check FPS in frontend (should be consistent)
   - Check latency in model server logs
   - Check CPU/RAM usage

4. **Add more cameras** (if needed):
   - Copy camera config in backend
   - Restart backend
   - Update frontend to show all cameras

---

## Quick Command Reference

### YOU (Model Server):
```bash
# Setup (one time)
cloudflared tunnel login
cloudflared tunnel create women-safety-model

# Run (daily)
.\start_model_server.bat
# OR
cloudflared tunnel run women-safety-model              # Terminal 1
python model_server.py                                 # Terminal 2

# Test
curl https://model.yourdomain.com/
```

### FRIEND (Backend):
```bash
# Setup (first time)
pip install -r requirements.txt

# Run
python -m uvicorn api.main:app --port 8000 --workers=1
```

### FRIEND (Frontend):
```bash
# Setup (first time)
cd dashboard
npm install

# Run
npm run dev
```

---

**You're all set! Start with terminal commands above and follow the checklist.** 🚀

