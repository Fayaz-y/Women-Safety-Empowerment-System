# Women Safety AI — Distributed Architecture Setup

## Architecture Overview

```
Your Laptop (Model Server)
├── Model Inference Server (port 9000)
│   ├── YOLO Detection
│   ├── Gender Classifier
│   ├── Assault Detector
│   ├── Pose Estimator
│   └── Proximity Engine
└── Cloudflare Tunnel
    └── https://model.yourdomain.com

Friend's Laptop (Backend + Frontend/Camera)
├── FastAPI Backend (port 8000)
│   └── Calls Model Server via https://model.yourdomain.com
├── Next.js Dashboard (port 3000)
└── Camera Capture
    └── Sends frames to backend for inference
```

---

## 🎯 Why This Architecture?

✅ **Separates concerns:**
- Your laptop: Heavy GPU computation (models)
- Friend's laptop: Application logic & user interface

✅ **Flexibility:**
- Friend can run backend/frontend anywhere
- You can upgrade models independently
- Models stay on your machine

✅ **Efficient:**
- Only inference data crosses the network (small image data)
- Friend's machine doesn't need GPU
- Your GPU stays dedicated to inference

---

## Step 1: Model Server Setup (YOUR Laptop)

### 1.1 Install Cloudflare Tunnel (Windows)

```bash
# Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
# Or: choco install cloudflared

cloudflared --version  # Verify
```

### 1.2 Authenticate Cloudflare

```bash
cloudflared tunnel login
# Approve in browser when prompted
```

### 1.3 Create Model Tunnel

```bash
cloudflared tunnel create women-safety-model
```

Save the **Tunnel ID** from output.

### 1.4 Configure Tunnel

Edit `C:\Users\<YourUsername>\.cloudflared\config.yml`:

```yaml
tunnel: women-safety-model
credentials-file: C:\Users\<YourUsername>\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: model.yourdomain.com
    service: http://localhost:9000
  - service: http_status:404
```

### 1.5 Configure Cloudflare DNS

In your **Cloudflare Dashboard** → **DNS**:

Add CNAME record:
- **Name**: `model`
- **Target**: `<tunnel-id>.cfargotunnel.com`
- **Proxy**: Orange (proxied)

Your model server URL: `https://model.yourdomain.com`

### 1.6 Update .env on YOUR Machine

```env
# Use local models (don't use remote)
USE_REMOTE_MODEL=false
DEVICE=cuda
```

This keeps the models loaded locally on your laptop.

### 1.7 Start Model Server

**Terminal 1 — Start Cloudflare Tunnel:**
```bash
cloudflared tunnel run women-safety-model
```

Keep this running. You'll see:
```
CONNECT   INF tunnel running at https://model.yourdomain.com
```

**Terminal 2 — Start Model Server:**
```bash
cd d:\@Women_safety\women_safety
python model_server.py
```

You should see:
```
[Model Server] Loading models...
[Model Server] Using device: cuda
[Model Server] ✓ All models loaded successfully!
```

### 1.8 Test Model Server

From any terminal:
```bash
# Test if accessible
curl https://model.yourdomain.com/

# Should return:
# {"service":"Women Safety AI — Model Inference Server","version":"1.0.0","models_loaded":"5/5","device":"cuda"}
```

---

## Step 2: Backend Setup (FRIEND's Laptop)

### 2.1 Clone/Copy the Code

Friend should have the backend code in a directory like `C:\women_safety\`.

### 2.2 Update .env File

**Friend's .env:**
```env
# ─── Important: Use Remote Model Server ───
USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
MODEL_API_TIMEOUT=30

# ─── Database (optional, can be local or remote) ───
DATABASE_URL=postgresql://postgres:password@localhost:5432/women_safety
REDIS_URL=redis://localhost:6379/0

# ─── Camera ───
CAMERA_TOGGLE=0  # 0=webcam, 1=USB cam, 2=video file
CAMERA_SOURCE_PATH=

# ─── API ───
API_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:3000

# ─── Keep rest default ───
DEVICE=cpu  # Not used since USE_REMOTE_MODEL=true
FUSION_THRESHOLD=0.75
# ... other settings
```

### 2.3 Start Backend (NO Cloudflare Needed)

Friend just runs the backend normally:

```bash
cd C:\women_safety
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

Backend will:
- Capture from camera on friend's machine
- Send frames to YOUR model server at `https://model.yourdomain.com`
- Get inference results back
- Process and store in local database

---

## Step 3: Frontend Setup (FRIEND's Laptop)

### 3.1 Configure Frontend .env.local

**Friend's dashboard/.env.local:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

(Same laptop, so both on localhost)

### 3.2 Start Frontend

```bash
cd dashboard
npm install  # If needed
npm run dev
```

Friend accesses at: `http://localhost:3000`

---

## Data Flow Example

**Sequence of events:**

```
1. Camera captures frame on Friend's laptop
   ↓
2. Backend receives frame
   ↓
3. Backend calls: POST https://model.yourdomain.com/api/v1/detect
   ↓
4. Your Model Server (GPU) processes instantly
   ↓
5. Returns detections back to Backend
   ↓
6. Backend analyzes and stores alert if needed
   ↓
7. Frontend displays on Friend's browser
```

---

## Environment Variables Summary

### Your Laptop (.env)
```env
USE_REMOTE_MODEL=false
MODEL_API_URL=http://localhost:9000  # Or could be remote
DEVICE=cuda
```

### Friend's Laptop (.env)
```env
USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
DEVICE=cpu  # Not used
```

---

## Troubleshooting

### ❌ Model server won't start
```bash
# Check if port 9000 is in use
netstat -ano | findstr :9000

# If in use, kill it:
taskkill /PID <PID> /F

# Or change port in config
```

### ❌ Backend can't reach model server
```bash
# Test from friend's laptop:
curl https://model.yourdomain.com/

# If fails:
# 1. Check YOUR model tunnel is running
# 2. Check DNS: nslookup model.yourdomain.com
# 3. Check firewall allows outbound HTTPS
```

### ❌ Cloudflare tunnel keeps disconnecting
```bash
# Increase verbosity:
cloudflared tunnel run women-safety-model --loglevel debug

# Check logs for errors like:
# - Invalid credentials
# - Wrong tunnel name
# - Port already in use
```

### ❌ CORS errors on backend
```
# If backend can't call model server, check:
# 1. MODEL_API_URL is correct in .env
# 2. Model server CORS allows backend IP
# (model_server.py has allow_origins=["*"])
```

### ❌ Model loads but no inference
```bash
# Check model server health:
curl https://model.yourdomain.com/api/v1/health

# Should return:
# {"status":"ok","models":{"detection":true,"assault":true,...},"device":"cuda"}
```

---

## Advanced: Auto-Start Scripts

### Windows: Model Server Auto-Start

Create `start_model_server.bat`:
```batch
@echo off
echo Starting Model Server...
start cmd /k "cloudflared tunnel run women-safety-model"
timeout /t 2
start cmd /k "python model_server.py"
```

### Windows: Backend Auto-Start (Friend)

Create `start_backend.bat`:
```batch
@echo off
echo Starting Backend on port 8000...
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

---

## Security Considerations

✅ **Currently:**
- Model server accessible at fixed URL
- No authentication on model endpoints

⚠️ **For Production:**

Add API key authentication to model server:

```python
# In model_server.py
from fastapi import Header

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=403)
    return x_api_key
```

Then add to .env:
```env
MODEL_API_KEY=your-secret-key
```

---

## Testing the Full System

### From Friend's Laptop:

1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/
   # Returns API info
   ```

2. **Check backend can reach model server:**
   ```bash
   curl https://model.yourdomain.com/
   # Returns model server info
   ```

3. **Check frontend loads:**
   - Open `http://localhost:3000` in browser
   - Should see login page

4. **Check inference works:**
   - Login and start monitoring
   - Ensure camera feed loads
   - Check browser console (F12) for errors

---

## Switching Between Local and Remote Models

### To use LOCAL models (debug on single laptop):
```env
USE_REMOTE_MODEL=false
MODEL_API_URL=http://localhost:9000
DEVICE=cuda
```

### To use REMOTE models (distributed):
```env
USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
DEVICE=cpu
```

Just change `.env` and restart backend!

---

## Next Steps

1. ✅ Set up model server on YOUR laptop with Cloudflare tunnel
2. ✅ Test at `https://model.yourdomain.com/`
3. ✅ Share URL with friend
4. ✅ Friend updates `.env` with `USE_REMOTE_MODEL=true`
5. ✅ Friend starts backend
6. ✅ Friend starts frontend
7. ✅ System works!

