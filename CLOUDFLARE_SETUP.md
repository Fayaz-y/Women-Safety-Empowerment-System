# Women Safety AI — Cloudflare Tunneling Setup Guide

This guide explains how to run the backend on your laptop, expose it via Cloudflare Tunnel, and have your friend run the frontend remotely.

---

## Architecture Overview

```
Your Laptop (Backend)
├── Python FastAPI (port 8000)
└── Cloudflare Tunnel
    └── https://women-safety.yourdomain.com

Friend's Computer (Frontend)
├── Next.js Dashboard (port 3000)
└── Connects to: https://women-safety.yourdomain.com
```

---

## Step 1: Install Cloudflare Tunnel

### Windows:
```bash
# Option A: Download from Cloudflare
# https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

# Option B: Using Chocolatey
choco install cloudflared
```

### Verify installation:
```bash
cloudflared --version
```

---

## Step 2: Authenticate Cloudflare

```bash
cloudflared tunnel login
```

This opens a browser window to authorize your account. Approve it and return to terminal.

---

## Step 3: Create a Tunnel

```bash
cloudflared tunnel create women-safety-backend
```

You'll get output like:
```
Tunnel credentials written to: C:\Users\<Username>\.cloudflared\abc123def456.json
Tunnel ID: abc123def456xyz789
```

**Save the Tunnel ID** — you'll need it next.

---

## Step 4: Configure Cloudflare DNS

1. Go to your **Cloudflare Dashboard** → **DNS**
2. Create a CNAME record:
   - **Name**: `women-safety` (or your chosen subdomain)
   - **Target**: `<TUNNEL_ID>.cfargotunnel.com` (e.g., `abc123def456xyz789.cfargotunnel.com`)
   - **Proxy**: Orange cloud (proxied)

Your public URL will be: `https://women-safety.yourdomain.com`

---

## Step 5: Configure the Tunnel (On Your Laptop)

Edit `C:\Users\<YourUsername>\.cloudflared\config.yml`:

```yaml
tunnel: women-safety-backend
credentials-file: C:\Users\<YourUsername>\.cloudflared\abc123def456.json

ingress:
  - hostname: women-safety.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

---

## Step 6: Update Backend Environment Variables

Edit your `.env` file:

**On YOUR laptop (running backend):**
```env
# API Configuration
API_URL=https://women-safety.yourdomain.com
ALLOWED_ORIGINS=http://localhost:3000,https://women-safety.yourdomain.com,https://dashboard.yourdomain.com

# Keep these for local development
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## Step 7: Run Backend with Cloudflare Tunnel

### Terminal 1 — Start Cloudflare Tunnel:
```bash
cloudflared tunnel run women-safety-backend
```

You'll see:
```
2026-04-02 10:30:00 CONNECT   INF tunnel running at https://women-safety.yourdomain.com
```

**Keep this running!**

### Terminal 2 — Start FastAPI Backend:
```bash
cd d:\@Women_safety\women_safety
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```

---

## Step 8: Frontend Configuration (Your Friend's Computer)

Your friend needs to:

### 1. Clone/have the dashboard code
```bash
cd dashboard
```

### 2. Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=https://women-safety.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://women-safety.yourdomain.com
```

### 3. Install and run:
```bash
npm install
npm run dev
```

Frontend will run on `http://localhost:3000` on their computer, but communicate with your Cloudflare tunnel backend.

---

## Testing the Connection

### Test from Friend's Computer:

**Browser Console:**
```javascript
// Should return user data if authenticated
fetch('https://women-safety.yourdomain.com/api/v1/cameras', {
  headers: { 'Authorization': 'Bearer YOUR_TOKEN' }
})
  .then(r => r.json())
  .then(console.log)
```

**Or from Terminal:**
```bash
# Test backend is accessible
curl https://women-safety.yourdomain.com/

# Should return: {"service":"Women Safety AI API","version":"1.0.0"}
```

---

## Troubleshooting

### ❌ Frontend gets CORS errors
- Check `.env` — `ALLOWED_ORIGINS` must include friend's frontend URL
- Restart backend after changing `.env`

### ❌ WebSocket connection fails (`wss://` not working)
- Cloudflare Tunnel supports WebSocket, but verify in Dashboard → Tunnels → Settings
- Ensure both `WEBSOCKET_ALLOWED` is enabled

### ❌ Tunnel command not found
- Reinstall: `choco uninstall cloudflared && choco install cloudflared`
- Or download from: https://github.com/cloudflare/cloudflared/releases

### ❌ Backend works locally but not over tunnel
- Check firewall: Windows Defender → Firewall → Allow app → Add `cloudflared.exe`
- Restart tunnel: `cloudflared tunnel run women-safety-backend`

---

## Local Development (No Tunnel)

If you want to test locally without Cloudflare:

**Your .env:**
```env
API_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

Then just run backend and frontend on `localhost`.

---

## Running with Cloudflare — Quick Summary

**You (Backend):**
1. `cloudflared tunnel run women-safety-backend` (Terminal 1)
2. `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` (Terminal 2)

**Friend (Frontend):**
1. Set `.env.local` with your Cloudflare URL
2. `npm run dev`
3. Access frontend at `http://localhost:3000`
4. Frontend securely connects to your backend via Cloudflare

---

## Advanced: Auto-Start Tunnel on Windows Boot

Create a `run-tunnel.bat`:
```batch
@echo off
cloudflared tunnel run women-safety-backend
pause
```

Create Windows Task Scheduler task to run this on startup.

---

## Security Notes

✅ **Cloudflare Tunnel provides:**
- End-to-end encryption
- DDoS protection
- No need to open ports on your router
- Automatic SSL/TLS certificate

⚠️ **Still needed:**
- Keep `JWT_SECRET_KEY` secure in `.env`
- Use environment-specific tokens
- Rotate secrets regularly

---

## Additional Resources

- Cloudflare Tunnel Docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
- FastAPI CORS: https://fastapi.tiangolo.com/tutorial/cors/
- Next.js Environment Variables: https://nextjs.org/docs/basic-features/environment-variables
