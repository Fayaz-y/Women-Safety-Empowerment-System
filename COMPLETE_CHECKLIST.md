# Complete Distributed Setup Checklist

## System Architecture

```
YOU (Model Server)          FRIEND (Backend + Frontend)
┌─────────────────┐        ┌──────────────────────────┐
│  Laptop w/ GPU  │        │  Laptop (any specs)      │
├─────────────────┤        ├──────────────────────────┤
│ Model Server    │◄──────►│ Backend (port 8000)      │
│ (port 9000)     │ HTTPS  │ Frontend (port 3000)     │
│ + Cloudflare    │        │ Camera Capture           │
│   Tunnel        │        │ Database (local/remote)  │
└─────────────────┘        └──────────────────────────┘
```

---

## ✅ YOUR CHECKLIST (Model Server)

### 1. Install Cloudflare Tunnel
- [ ] Download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
- [ ] Or: `choco install cloudflared`
- [ ] Verify: `cloudflared --version`

### 2. Cloudflare Setup
- [ ] Run: `cloudflared tunnel login`
- [ ] Approve in browser
- [ ] Run: `cloudflared tunnel create women-safety-model`
- [ ] Save the **Tunnel ID**

### 3. Configure Tunnel
- [ ] Edit: `C:\Users\<YourUsername>\.cloudflared\config.yml`
  ```yaml
  tunnel: women-safety-model
  credentials-file: C:\Users\<YourUsername>\.cloudflared\<TUNNEL_ID>.json
  
  ingress:
    - hostname: model.yourdomain.com
      service: http://localhost:9000
    - service: http_status:404
  ```

### 4. Configure Cloudflare DNS
- [ ] Go to Cloudflare Dashboard → DNS
- [ ] Add CNAME record:
  - [ ] Name: `model`
  - [ ] Target: `<TUNNEL_ID>.cfargotunnel.com`
  - [ ] Proxy: Orange (proxied)
  - [ ] DNS entry created

### 5. Prepare Your Laptop
- [ ] Model files in `./models/` directory
- [ ] `.env` file has: `USE_REMOTE_MODEL=false`
- [ ] `.env` file has: `DEVICE=cuda` (or cpu)

### 6. Start Model Server
- [ ] Terminal 1: `cloudflared tunnel run women-safety-model`
  - [ ] Confirm: "tunnel running at https://model.yourdomain.com"
  - [ ] Keep this running
  
- [ ] Terminal 2: `python model_server.py`
  - [ ] Confirm: "[Model Server] ✓ All models loaded successfully!"
  - [ ] Keep this running

### 7. Test Model Server
- [ ] Run in terminal: `curl https://model.yourdomain.com/`
- [ ] Should return: `{"service":"Women Safety AI — Model Inference Server"...}`
- [ ] ✅ Your part is done!

### 8. Share with Friend
- [ ] Give them: `https://model.yourdomain.com` (your model server URL)
- [ ] Tell them to set in `.env`: `MODEL_API_URL=https://model.yourdomain.com`
- [ ] Keep both terminals running!

---

## ✅ FRIEND'S CHECKLIST (Backend + Frontend)

### 1. Get the Code
- [ ] Copy Women Safety project to their laptop
- [ ] Or: `git clone <repository>`
- [ ] Directory: `C:\women_safety\` (or wherever)

### 2. Install Dependencies
- [ ] Terminal: `pip install -r requirements.txt`
- [ ] Then: `cd dashboard && npm install && cd ..`

### 3. Configure Environment
- [ ] Create/edit `.env` in project root
- [ ] **CRITICAL**: Set these
  ```env
  USE_REMOTE_MODEL=true
  MODEL_API_URL=https://model.yourdomain.com
  ```
- [ ] Set database URL (local or remote)
- [ ] Set `CAMERA_TOGGLE=0` (or 1 for USB camera)
- [ ] Copy other settings from template

### 4. Create Database (if using local PostgreSQL)
- [ ] Run: `python scripts/create_tables.py`
- [ ] Or: `python scripts/seed_db.py`

### 5. Start Backend
- [ ] Terminal 1: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1`
- [ ] Confirm: "Uvicorn running on http://0.0.0.0:8000"
- [ ] Keep this running

### 6. Start Frontend
- [ ] Terminal 2: `cd dashboard && npm run dev`
- [ ] Confirm: "Local: http://localhost:3000"
- [ ] Keep this running

### 7. Test Dashboard
- [ ] Open browser: `http://localhost:3000`
- [ ] Should see login page
- [ ] Frontend working ✓

### 8. Test Backend Connection
- [ ] Try logging in
- [ ] Check browser console (F12) for errors
- [ ] If no errors: Backend ↔ Model Server working ✓

### 9. Test Inference
- [ ] Allow camera access when prompted
- [ ] Dashboard should load camera feed
- [ ] Models should be processing frames
- [ ] Alerts (if any) should appear
- [ ] System working ✓

---

## Verification Steps

### From Your Laptop:
```bash
# Test model server is accessible
curl https://model.yourdomain.com/
# Returns: {"service":"Women Safety AI — Model Inference Server",...}

# Check health
curl https://model.yourdomain.com/api/v1/health
# Returns: {"status":"ok","models":{"detection":true,...},"device":"cuda"}
```

### From Friend's Laptop:
```bash
# Test backend is running
curl http://localhost:8000/
# Returns: {"service":"Women Safety AI API","version":"1.0.0"}

# Test backend can reach model server
# (This should work if configured correctly)
# Check backend terminal or logs for errors
```

### In Browser (Friend's):
```javascript
// Open http://localhost:3000/
// Press F12 (Developer Tools)
// Go to Console tab
// Paste:

fetch('http://localhost:8000/')
  .then(r => r.json())
  .then(console.log)
  .catch(e => console.error('Backend error:', e))

// Should print API info, no CORS errors
```

---

## Troubleshooting Matrix

| Problem | Check | Solution |
|---------|-------|----------|
| Model server won't start | Port 9000 in use | `taskkill /PID <PID> /F` |
| Tunnel disconnects | Cloudflare auth | `cloudflared tunnel login` again |
| Friend can't reach models | URL wrong in .env | Verify `MODEL_API_URL` exactly |
| Backend can't load database | DB not running | Start PostgreSQL/Redis |
| Frontend shows blank | Backend not running | Start terminal 1 on friend's side |
| Camera not working | CAMERA_TOGGLE wrong | Check value (0/1/2) |
| CORS error | Origins not allowed | Backend .env ALLOWED_ORIGINS |
| Slow inference | Network latency | Normal with remote models |

---

## Environment Variables Quick Reference

### YOUR .env (Model Server)
```env
# Must be false (models loaded locally)
USE_REMOTE_MODEL=false

# Can be local or http://localhost:9000
MODEL_API_URL=http://localhost:9000

# GPU acceleration
DEVICE=cuda
USE_FP16=true
```

### FRIEND's .env (Backend + Frontend)
```env
# Must be true (use your model server)
USE_REMOTE_MODEL=true

# Your model server URL
MODEL_API_URL=https://model.yourdomain.com

# Database (local or remote)
DATABASE_URL=postgresql://localhost:5432/women_safety
REDIS_URL=redis://localhost:6379/0

# Camera on friend's machine
CAMERA_TOGGLE=0

# Frontend should connect to backend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Don't use GPU (not available)
DEVICE=cpu
```

---

## Ports Reference

| Port | Service | Machine | Can Close? |
|------|---------|---------|-----------|
| 9000 | Model API | YOUR | No (backend depends on it) |
| 8000 | Backend | FRIEND | No (frontend depends on it) |
| 3000 | Frontend | FRIEND | Yes (can reopen) |
| 443 | HTTPS Tunnel | Internet | No (tunnel must be running) |
| 5432 | PostgreSQL | Depends | No (if backend uses it) |
| 6379 | Redis | Depends | No (if backend uses it) |

---

## Data Flow Example

```
1. Camera captures frame on Friend's laptop
   ↓
2. Friend's backend reads frame
   ↓
3. Backend calls: POST https://model.yourdomain.com/api/v1/detect
   ↓ (Frame travels via HTTPS, encrypted)
4. Your model server receives request
   ↓
5. Your GPU runs YOLO detection model
   ↓
6. Returns: {"detections": [{"box": [...], "confidence": 0.95}]}
   ↓ (Results travel back via HTTPS)
7. Friend's backend receives response
   ↓
8. Backend analyzes detections
   ↓
9. If alert → Stores in database
   ↓
10. Frontend polls database via WebSocket
   ↓
11. Dashboard updates in real-time
```

**Latency:**
- Model inference: 50-200ms (depends on GPU)
- Network round trip: 10-50ms (depends on internet)
- **Total per frame: 100-300ms** (very acceptable!)

---

## Next Steps After Setup

1. **Test with static image** (verify model works)
2. **Test with webcam** (verify camera works)
3. **Test with different scenarios** (dim light, crowded, etc.)
4. **Monitor performance** (check inference speed)
5. **Set thresholds** if needed (FUSION_THRESHOLD, etc.)

---

## Stopping the System

### Your Laptop:
1. Press Ctrl+C in model_server.py window
2. Press Ctrl+C in cloudflared tunnel window
3. Confirm both closed

### Friend's Laptop:
1. Press Ctrl+C in frontend window (npm run dev)
2. Press Ctrl+C in backend window (uvicorn)
3. Confirm both closed

---

## Restarting the System

### Morning (Start):
1. YOU: Run `start_model_server.bat` (or two terminals)
2. FRIEND: Run backend (terminal 1)
3. FRIEND: Run frontend (terminal 2)

### Evening (Stop):
1. FRIEND: Close both terminals
2. YOU: Close both terminals

---

## Tips for Success

✅ **DO:**
- Keep both setup documents (your: DISTRIBUTED_SETUP.md, friend's: FRIEND_SETUP.md)
- Test model server before backend connects
- Use `curl` to debug network issues
- Check browser console (F12) for frontend errors
- Monitor backend terminal for Python errors

❌ **DON'T:**
- Close terminals unexpectedly (can cause stale connections)
- Change `.env` while system is running (restart backend after)
- Use different model server URLs by mistake
- Run model server on multiple machines (will conflict)

---

## Support & Debugging

1. **Check all three services running:**
   - Model server (your laptop)
   - Backend (friend's laptop)
   - Frontend (friend's laptop)

2. **Check logs:**
   - Model server: Terminal where you ran `python model_server.py`
   - Backend: Terminal where you ran `python -m uvicorn`
   - Frontend: Browser console (F12)

3. **Test connectivity:**
   ```bash
   # From friend's laptop:
   ping model.yourdomain.com
   curl https://model.yourdomain.com/
   ```

4. **Isolate the problem:**
   - Model server failing? → Check GPU
   - Backend failing? → Check database
   - Frontend failing? → Check browser console
   - Models not processing? → Check MODEL_API_URL

---

## Success Criteria

✅ **System is working when:**
- [ ] Model server running on your laptop
- [ ] Friend's backend running
- [ ] Friend's frontend displaying at localhost:3000
- [ ] Frontend can log in
- [ ] Camera feed loads
- [ ] Inference runs (check latency is reasonable)
- [ ] Alerts trigger when needed

**Congratulations! You're done!** 🎉

