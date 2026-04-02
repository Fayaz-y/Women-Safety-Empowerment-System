# 🚀 NGROK QUICK START (Super Simple!)

**This is the EASIEST way to run everything.**

---

## Installation (1 minute)

```bash
choco install ngrok
```

Or download from: https://ngrok.com/download

---

## Setup Account (2 minutes)

1. Go to: https://ngrok.com
2. Sign up with email
3. Check email, verify
4. Go to dashboard, copy **Auth Token**
5. Run in terminal:
   ```bash
   ngrok config add-authtoken <PASTE_TOKEN_HERE>
   ```

Done! ✅

---

## YOUR LAPTOP (Model Server)

### Terminal 1️⃣
```bash
ngrok http 9000
```

You'll see:
```
Forwarding     https://abc123def456.ngrok.io -> http://localhost:9000
```

**Copy this URL:** `https://abc123def456.ngrok.io`

### Terminal 2️⃣
```bash
cd d:\@Women_safety\women_safety
python model_server.py
```

Wait for: `[Model Server] ✓ All models loaded successfully!`

---

## FRIEND'S LAPTOP

### Update `.env`
Replace:
```env
MODEL_API_URL=https://abc123def456.ngrok.io
```

(Use the URL you copied above)

### Terminal 1️⃣
```bash
python -m uvicorn api.main:app --port 8000
```

### Terminal 2️⃣
```bash
cd dashboard
npm run dev
```

---

## Open Dashboard

```
http://localhost:3000
```

✅ **Done! Everything works!**

---

## When You Restart

Each time you restart ngrok:
1. `ngrok http 9000`
2. Get NEW URL
3. Send to friend
4. Friend updates `.env`
5. Friend restarts backend

Takes 30 seconds. That's it!

---

## Test It

```bash
curl https://abc123def456.ngrok.io/
# Should work instantly
```

---

## Summary

| Step | Command | Time |
|------|---------|------|
| Install | `choco install ngrok` | 1 min |
| Setup token | `ngrok config add-authtoken <token>` | 2 min |
| Run tunnel | `ngrok http 9000` | 5 sec |
| Run model | `python model_server.py` | 2 min |
| Tell friend URL | Copy/paste | 10 sec |
| **Total** | | **5-6 minutes!** |

✅ **SO MUCH EASIER than Cloudflare!**

---

**That's it! You're done! ngrok is the way to go.** 🎉
