# Quick Start — Distributed Setup

## Your Role (Model Server)

You have the machine with a **GPU** and will run the inference models.

### What you need to do:

1. **Install Cloudflare Tunnel**
   ```bash
   # Windows: choco install cloudflared
   # Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
   ```

2. **Authenticate**
   ```bash
   cloudflared tunnel login
   ```

3. **Create tunnel**
   ```bash
   cloudflared tunnel create women-safety-model
   ```
   Save the Tunnel ID

4. **Configure DNS in Cloudflare Dashboard**
   - Add CNAME: `model` → `<tunnel-id>.cfargotunnel.com`

5. **Edit config file at `C:\Users\<YourUsername>\.cloudflared\config.yml`:**
   ```yaml
   tunnel: women-safety-model
   credentials-file: C:\Users\<YourUsername>\.cloudflared\<tunnel-id>.json
   
   ingress:
     - hostname: model.yourdomain.com
       service: http://localhost:9000
     - service: http_status:404
   ```

6. **Run the startup script**
   ```bash
   .\start_model_server.bat
   ```

   This opens two windows:
   - Cloudflare Tunnel (keep running)
   - Model Server (loads models, listens on 9000)

7. **Test it works**
   ```bash
   curl https://model.yourdomain.com/
   
   # Should return:
   # {"service":"Women Safety AI — Model Inference Server","version":"1.0.0",...}
   ```

8. **Share with your friend:**
   - Give them the model URL: `https://model.yourdomain.com`
   - Keep these windows running!

---

## Your Friend's Role (Backend + Frontend)

Your friend runs the application backend and dashboard.

### What your friend needs to do:

1. **Get the code**
   - Copy the project to their laptop
   - Minimum: `api/`, `core/`, `config/`, `.env`

2. **Update their `.env` file:**
   ```env
   # CRITICAL: Tell backend to use YOUR model server
   USE_REMOTE_MODEL=true
   MODEL_API_URL=https://model.yourdomain.com
   MODEL_API_TIMEOUT=30
   
   # Database (can be local or remote)
   DATABASE_URL=postgresql://...
   REDIS_URL=redis://...
   
   # Camera on their machine
   CAMERA_TOGGLE=0
   
   # Keep these
   DEVICE=cpu
   FUSION_THRESHOLD=0.75
   # ... other settings
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
   ```

5. **Start the frontend** (in another terminal, in `dashboard/` folder)
   ```bash
   npm install
   npm run dev
   ```

6. **Access the system**
   - Frontend: `http://localhost:3000`
   - Backend: `http://localhost:8000`
   - Model Server: `https://model.yourdomain.com` (through your laptop)

---

## Data Flow

```
Friend's Camera
    ↓
Friend's Backend (port 8000)
    ↓ (sends frames via HTTPS)
YOUR Model Server (https://model.yourdomain.com)
    ↓ (returns detection results)
Friend's Backend (processes, stores results)
    ↓ (WebSocket)
Friend's Dashboard (http://localhost:3000)
    ↓ (displays to screen)
```

---

## Files You Need to Know

| File | Purpose |
|------|---------|
| `model_server.py` | The inference server you run |
| `core/remote_model.py` | How backend calls your models |
| `.env` | Configuration (your: `USE_REMOTE_MODEL=false`, friend's: `USE_REMOTE_MODEL=true`) |
| `start_model_server.bat` | One-click startup for you |

---

## Troubleshooting

### Model Server Won't Start
```bash
# Check port 9000 is free
netstat -ano | findstr :9000

# Kill if needed
taskkill /PID <PID> /F
```

### Friend Can't Reach Model Server
```bash
# From friend's laptop, test:
curl https://model.yourdomain.com/

# If fails, check:
# 1. Your model tunnel is running
# 2. DNS is propagated (nslookup model.yourdomain.com)
# 3. Your firewall allows outbound HTTPS on port 443
```

### Backend Says "Model API Failed"
```
# In friend's backend logs, if you see:
# "Failed to connect to model server"
# 
# Check:
# 1. MODEL_API_URL in friend's .env is correct
# 2. Models are loaded on your server (check your model_server logs)
# 3. Network connectivity between laptops
```

---

## Summary

| Machine | Runs | Keeps Running |
|---------|------|--|
| **Your Laptop** | Model Server + Tunnel | 2 terminals |
| **Friend's Laptop** | Backend + Frontend | 2 terminals |

That's it! The two communicate automatically via HTTPS through Cloudflare.

