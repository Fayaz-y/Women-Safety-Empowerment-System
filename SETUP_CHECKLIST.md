# Women Safety AI — Cloudflare Tunneling Setup Checklist

## 🎯 Overview
Backend runs on your laptop, exposes via Cloudflare Tunnel, friend accesses frontend remotely.

---

## 👨‍💻 BACKEND SETUP (Your Laptop)

### Prerequisites
- [ ] Python 3.10+ installed  
- [ ] `pip install -r requirements.txt` completed
- [ ] PostgreSQL & Redis running locally
- [ ] `.env` file configured

### Cloudflare Tunnel Setup

#### 1. Install Cloudflare Tunnel
- [ ] Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
- [ ] Or: `choco install cloudflared`
- [ ] Verify: Run `cloudflared --version`

#### 2. Authenticate
- [ ] Run: `cloudflared tunnel login`
- [ ] Approve in browser
- [ ] Credentials saved to `C:\Users\<Username>\.cloudflared\`

#### 3. Create Tunnel
- [ ] Run: `cloudflared tunnel create women-safety-backend`
- [ ] **Save the Tunnel ID** from output
- [ ] Credential file created at `~/.cloudflared/<tunnel-id>.json`

#### 4. Configure Tunnel
- [ ] Edit `C:\Users\<Username>\.cloudflared\config.yml`:
  ```yaml
  tunnel: women-safety-backend
  credentials-file: C:\Users\<Username>\.cloudflared\<tunnel-id>.json
  ingress:
    - hostname: women-safety.yourdomain.com
      service: http://localhost:8000
    - service: http_status:404
  ```

#### 5. Configure Cloudflare DNS
- [ ] Log in to Cloudflare Dashboard
- [ ] Go to DNS settings
- [ ] Add CNAME record:
  - Name: `women-safety`
  - Target: `<tunnel-id>.cfargotunnel.com`
  - Proxy: Orange cloud (proxied)
- [ ] Wait for DNS propagation (usually instant)

#### 6. Update Backend Configuration
- [ ] Edit `.env`:
  ```env
  API_URL=https://women-safety.yourdomain.com
  ALLOWED_ORIGINS=http://localhost:3000,https://women-safety.yourdomain.com
  ```
- [ ] Verify settings loaded: Backend automatically uses these on startup

#### 7. Start Backend
- [ ] **Terminal 1 - Start Cloudflare Tunnel:**
  ```bash
  cloudflared tunnel run women-safety-backend
  ```
  - [ ] Verify output shows: `tunnel running at https://women-safety.yourdomain.com`
  - [ ] **Keep this running!**

- [ ] **Terminal 2 - Start FastAPI:**
  ```bash
  cd d:\@Women_safety\women_safety
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
  ```
  - [ ] Verify output shows: `Uvicorn running on http://0.0.0.0:8000`

#### 8. Test Backend
- [ ] Open browser: `https://women-safety.yourdomain.com/`
- [ ] Should see: `{"service":"Women Safety AI API","version":"1.0.0"}`
- [ ] If fails → Check:
  - [ ] Cloudflare tunnel is running
  - [ ] FastAPI is running on 8000
  - [ ] DNS is propagated (`nslookup women-safety.yourdomain.com`)

---

## 👥 FRONTEND SETUP (Friend's Computer)

### Prerequisites
- [ ] Node.js 18+ installed
- [ ] `npm install` completed in `dashboard/` folder
- [ ] Backend Cloudflare URL from you: `https://women-safety.yourdomain.com`

### Configuration

#### 1. Get Backend Domain
- [ ] Ask you for your Cloudflare domain
- [ ] Should be: `https://women-safety.yourdomain.com`

#### 2. Create .env.local
- [ ] In `dashboard/` folder, create `.env.local` file:
  ```env
  NEXT_PUBLIC_API_URL=https://women-safety.yourdomain.com
  NEXT_PUBLIC_WS_URL=wss://women-safety.yourdomain.com
  ```

#### 3. Install Dependencies
- [ ] From `dashboard/` folder:
  ```bash
  npm install
  ```

#### 4. Start Frontend
- [ ] Run:
  ```bash
  npm run dev
  ```
- [ ] Verify output shows: `Local: http://localhost:3000`
- [ ] **Do NOT close this terminal**

#### 5. Test Frontend
- [ ] Open browser: `http://localhost:3000`
- [ ] Should be redirected to login
- [ ] Try logging in with test credentials
- [ ] Check browser console (F12) for CORS errors
  - [ ] If CORS error → **Tell backend user** so they can update `.env`

---

## 🧪 Testing Connection

### From Friend's Computer:

#### Visual Test
- [ ] Frontend loads at `http://localhost:3000`
- [ ] Click "Login"
- [ ] Page doesn't crash → Backend is reachable

#### Console Test (F12 → Console)
```javascript
// Should return cameras list or 401 if not authenticated
fetch('https://women-safety.yourdomain.com/api/v1/cameras')
  .then(r => r.json())
  .then(console.log)
  .catch(e => console.error('Error:', e))
```

#### Terminal Test  
```bash
# Test from friend's command line
curl https://women-safety.yourdomain.com/
# Should return: {"service":"Women Safety AI API","version":"1.0.0"}
```

---

## ⚠️ Troubleshooting

### ❌ Frontend shows CORS error
- **Cause:** Backend doesn't allow frontend origin
- **Fix:** Backend user must edit `.env`:
  ```env
  ALLOWED_ORIGINS=http://localhost:3000,https://women-safety.yourdomain.com,YOUR_FRIENDS_FRONTEND_URL
  ```
- **Then:** Restart FastAPI (Terminal 2)

### ❌ WebSocket fails (real-time alerts don't work)
- **Cause:** Need `wss://` for secure WebSocket
- **Check:** `.env.local` on frontend has `NEXT_PUBLIC_WS_URL=wss://...`
- **Fix:** Cloudflare Tunnel supports WebSocket by default

### ❌ `ERR_TOO_MANY_REDIRECTS`
- **Cause:** SSL/HTTPS loop
- **Fix:** Ensure frontend uses `https://` and `wss://` (not `http://` or `ws://`)

### ❌ `cloudflared: command not found`
- **Cause:** Not installed or not in PATH
- **Fix:** 
  - Download manually: https://github.com/cloudflare/cloudflared/releases
  - Extract to folder and add to PATH
  - Or: `choco uninstall cloudflared && choco install cloudflared`

### ❌ Tunnel won't start (`tunnel not found`)
- **Cause:** Tunnel doesn't exist or wrong name
- **Fix:** Run `cloudflared tunnel list` to see all tunnels
- **Then:** `cloudflared tunnel run <tunnel-name>`

### ❌ Backend works locally but not over tunnel
- **Cause:** Network issue or firewall
- **Check:**
  - [ ] Windows Firewall: Allow `cloudflared.exe`
  - [ ] Router: No port blocking
  - [ ] Run: `cloudflared tunnel info women-safety-backend`

### ❌ Database/Redis not found
- **Cause:** Running on localhost but expecting different host
- **Check:** `.env` has correct:
  - [ ] `DATABASE_URL=postgresql://...@localhost:...`
  - [ ] `REDIS_URL=redis://localhost:...`

---

## 🚀 Quick Start Scripts

### Backend (Windows)
```bash
# Use this shortcut to start everything
.\start_backend_cloudflare.bat
```

### Frontend (Windows)
```bash
cd dashboard
.\start_frontend_cloudflare.bat
# It will ask for backend domain
```

### Frontend (macOS/Linux)
```bash
cd dashboard
chmod +x start_frontend_cloudflare.sh
./start_frontend_cloudflare.sh
```

---

## ✅ Final Checklist

### Backend Ready For Friend?
- [ ] Cloudflare tunnel running
- [ ] FastAPI running on 8000
- [ ] `.env` has correct `ALLOWED_ORIGINS`
- [ ] Backend accessible at `https://women-safety.yourdomain.com/`
- [ ] WebSocket support enabled (automatic with Cloudflare)

### Friend Can Access Frontend?
- [ ] `.env.local` created with correct backends URL
- [ ] Frontend running on `http://localhost:3000`
- [ ] Can load login page
- [ ] No CORS errors in browser console
- [ ] Real-time features work (alerts, stream status)

---

## 📞 If Still Having Issues

1. **Check connectivity:**
   ```bash
   ping women-safety.yourdomain.com
   nslookup women-safety.yourdomain.com
   curl https://women-safety.yourdomain.com/
   ```

2. **Check logs:**
   - Cloudflare Tunnel: Look at terminal where tunnel is running
   - FastAPI: Look at terminal where API is running
   - Frontend: F12 → Console tab

3. **Share logs with each other** and compare with this guide

