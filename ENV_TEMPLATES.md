# Environment Variable Templates

## Copy and paste these into your `.env` files

---

## Template 1: YOUR .env (Model Server - Your Laptop with GPU)

```env
# ═══════════════════════════════════════════════════════════════════
# Women Safety AI — Model Inference Server Configuration
# YOU RUN THIS (on your laptop with GPU)
# ═══════════════════════════════════════════════════════════════════

# Database (optional, usually not needed for model server)
DATABASE_URL=postgresql://postgres:password@localhost:5432/women_safety
REDIS_URL=redis://localhost:6379/0

# ─── Device / Inference ───
# Keep these as-is; models run locally on your GPU
DEVICE=cuda
USE_FP16=true
MODEL_DIR=./models
SNAPSHOT_DIR=./data/snapshots
CLIP_DIR=./data/clips

# Detection Thresholds (not used by model server but keep for consistency)
FUSION_THRESHOLD=0.75
PROXIMITY_RADIUS_PX=150
ISOLATION_SECONDS=5.0
ALERT_COOLDOWN_SECONDS=30

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
ALERT_PHONE_NUMBER=

# JWT Auth
JWT_SECRET_KEY=change-me-in-production-minimum-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# ─── API / Networking ───
# Model Server doesn't call itself
API_URL=http://localhost:9000
ALLOWED_ORIGINS=*

# ─── Model Inference (Remote) ───
# CRITICAL: Keep these DISABLED on your machine
USE_REMOTE_MODEL=false
MODEL_API_URL=http://localhost:9000
MODEL_API_TIMEOUT=30

# Frontend URLs (not used by model server)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# ─── Camera ───
# Not used by model server, but keep defaults
CAMERA_TOGGLE=0
CAMERA_SOURCE_PATH=
```

---

## Template 2: FRIEND's .env (Backend + Frontend - Their Laptop)

```env
# ═══════════════════════════════════════════════════════════════════
# Women Safety AI — Backend & Frontend Configuration
# FRIEND RUNS THIS (on their laptop, no GPU needed)
# ═══════════════════════════════════════════════════════════════════

# ─── Database ───
# If PostgreSQL is on same laptop:
DATABASE_URL=postgresql://postgres:password@localhost:5432/women_safety
REDIS_URL=redis://localhost:6379/0

# Or if database is on your computer (the one with GPU):
# DATABASE_URL=postgresql://postgres:password@<YOUR_IP>:5432/women_safety
# REDIS_URL=redis://<YOUR_IP>:6379/0

# ─── Device / Inference ───
# NOT USED - backend will call remote models
DEVICE=cpu
USE_FP16=true
MODEL_DIR=./models
SNAPSHOT_DIR=./data/snapshots
CLIP_DIR=./data/clips

# Detection Thresholds
FUSION_THRESHOLD=0.75
PROXIMITY_RADIUS_PX=150
ISOLATION_SECONDS=5.0
ALERT_COOLDOWN_SECONDS=30

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
ALERT_PHONE_NUMBER=

# JWT Auth
JWT_SECRET_KEY=change-me-in-production-minimum-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# ═══════════════════════════════════════════════════════════════════
# *** CRITICAL: Remote Model Configuration ***
# ═══════════════════════════════════════════════════════════════════
# MUST be true - tells backend to use your model server
USE_REMOTE_MODEL=true

# MUST be set to your model server URL
# Example: https://model.yourdomain.com
# Ask your friend (the one with GPU) for their exact URL
MODEL_API_URL=https://model.yourdomain.com

# Timeout for waiting for model inference (in seconds)
# Increase if your network is slow
MODEL_API_TIMEOUT=30

# ─── API / Networking ───
# Backend URL (local, backend runs on same machine)
API_URL=http://localhost:8000

# CORS: Allow frontend on same machine
ALLOWED_ORIGINS=http://localhost:3000

# Frontend URLs (both run on same machine)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# ─── Camera ───
# 0 = built-in webcam
# 1 = external USB camera
# 2 = video file (set CAMERA_SOURCE_PATH)
CAMERA_TOGGLE=0
CAMERA_SOURCE_PATH=
```

---

## Template 3: FRIEND's dashboard/.env.local (Frontend Only)

```env
# Frontend environment variables
# This file tells the frontend where to find the backend API

# Backend is on same laptop
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## Quick Reference: What Changes?

| Setting | Your Laptop | Friend's Laptop |
|---------|-------------|-----------------|
| `USE_REMOTE_MODEL` | `false` | `true` |
| `MODEL_API_URL` | HTTP localhost | HTTPS Cloudflare URL |
| `DEVICE` | `cuda` | `cpu` |
| `DATABASE_URL` | (optional) | Required |
| `ALLOWED_ORIGINS` | `*` | `http://localhost:3000` |
| `CAMERA_TOGGLE` | N/A | 0 or 1 |
| `NEXT_PUBLIC_API_URL` | N/A | `http://localhost:8000` |

---

## Fill-in Template: Personalized for Your Setup

**Replace these placeholders with your actual values:**

```env
# === YOUR LAPTOP ===

USE_REMOTE_MODEL=false
DEVICE=cuda
MODEL_API_URL=http://localhost:9000

# === FRIEND'S LAPTOP ===

USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
DATABASE_URL=postgresql://postgres:PASSWORD@FRIEND_IP:5432/women_safety
REDIS_URL=redis://FRIEND_IP:6379/0
CAMERA_TOGGLE=0 (or 1 for USB)
```

---

## Validation Checklist

After setting `.env`, verify before running:

### Your Laptop:
- [ ] `USE_REMOTE_MODEL=false` ✓
- [ ] `DEVICE=cuda` or `cpu` ✓
- [ ] Models exist in `./models/` directory ✓

### Friend's Laptop:
- [ ] `USE_REMOTE_MODEL=true` ✓
- [ ] `MODEL_API_URL=https://model.yourdomain.com` (your correct URL) ✓
- [ ] `DATABASE_URL` points to valid PostgreSQL ✓
- [ ] `REDIS_URL` points to valid Redis ✓
- [ ] `CAMERA_TOGGLE` matches their camera setup ✓
- [ ] `NEXT_PUBLIC_API_URL=http://localhost:8000` ✓

---

## Common Mistakes to Avoid

❌ **DON'T:**
- Set `USE_REMOTE_MODEL=true` on your laptop
- Set `USE_REMOTE_MODEL=false` on friend's laptop
- Use `http://` instead of `https://` for Cloudflare URL
- Forget the `s` in `https://`
- Use wrong `MODEL_API_URL` (copy from your friend carefully!)
- Keep old `.env.bak` files that might override settings

✅ **DO:**
- Double-check `MODEL_API_URL` matches exactly
- Test with `curl https://model.yourdomain.com/` first
- Keep both `.env` files separate (one per laptop)
- Restart backend after changing `.env`

---

## Testing with curl

### Your Laptop:
```bash
# Test model server is responding
curl https://model.yourdomain.com/

# Check health
curl https://model.yourdomain.com/api/v1/health
```

### Friend's Laptop:
```bash
# Test backend is responding
curl http://localhost:8000/

# Test backend can reach your model server
# (Check backend logs for success/failure)
```

---

## If You're Not Sure

**Ask yourself:**
1. Which laptop am I on? → Use corresponding template
2. Do I have GPU access here? → If yes: `DEVICE=cuda, USE_REMOTE_MODEL=false`
3. Am I running models here? → If yes: `USE_REMOTE_MODEL=false`
4. Am I calling someone else's models? → If yes: `USE_REMOTE_MODEL=true`

---

That's it! Copy the template for YOUR setup, make necessary changes, and you're ready to go.

