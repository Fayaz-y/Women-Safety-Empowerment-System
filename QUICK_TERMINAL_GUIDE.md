# 🚀 QUICK START: 3-Terminal Command Sheet

Print this or pin it next to your screen while running!

---

## YOUR LAPTOP (Model Server - You)

### Terminal 1️⃣ (Cloudflare Tunnel)
```bash
cloudflared tunnel run women-safety-model
```
✅ **Wait for:** `tunnel running at https://model.yourdomain.com`

---

### Terminal 2️⃣ (Model Server)
```bash
cd d:\@Women_safety\women_safety
python model_server.py
```
✅ **Wait for:** `[Model Server] ✓ All models loaded successfully!`

---

---

## FRIEND'S LAPTOP (Backend + Frontend - Your Friend)

### Terminal 1️⃣ (Backend)

**First, edit `.env` file and change:**
```
USE_REMOTE_MODEL=true
MODEL_API_URL=https://model.yourdomain.com
```

**Then run:**
```bash
cd path\to\women_safety
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
```
✅ **Wait for:** `Uvicorn running on http://0.0.0.0:8000`

---

### Terminal 2️⃣ (Frontend)
```bash
cd path\to\women_safety\dashboard
npm install
npm run dev
```
✅ **Wait for:** `Local: http://localhost:3000`

---

---

## 🌐 ACCESS THE DASHBOARD

**Open browser on friend's laptop:**
```
http://localhost:3000
```

---

## ⚙️ OPTIONAL: Check If Everything Is Running

### From any terminal:
```bash
# Check model server
curl https://model.yourdomain.com/

# Check backend
curl http://localhost:8000/

# Check frontend
open http://localhost:3000
```

---

## ❌ QUICK FIXES

| Problem | Fix |
|---------|-----|
| Port in use | `lsof -i :8000` or `netstat -ano \| findstr :8000` |
| Python not found | Use full path: `C:\Python310\python.exe ...` |
| npm not found | Install Node.js from nodejs.org |
| Camera error | Click "Allow" when browser asks for camera |
| Model error | Check URL in `.env` matches exactly |

---

## 🛑 STOP EVERYTHING

Press **Ctrl+C** in each terminal in reverse order:
1. Terminal 2️⃣ Frontend (Ctrl+C)
2. Terminal 1️⃣ Backend (Ctrl+C)
3. Terminal 2️⃣ Model Server (Ctrl+C)
4. Terminal 1️⃣ Cloudflare (Ctrl+C)

---

## ⏱️ TOTAL TIME

- Model server startup: 30-90 seconds (loading models)
- Backend startup: 5-10 seconds
- Frontend startup: 10-15 seconds
- **Total:** ~2-3 minutes

---

## 📱 DASHBOARD URL

```
http://localhost:3000
```

Default login: `test@example.com` / `password` (or check database)

---

## 🎯 SUCCESS CHECKLIST

- [ ] 4 terminals running (2 on your laptop, 2 on friend's)
- [ ] Model server shows "models loaded"
- [ ] Backend shows "Uvicorn running"
- [ ] Frontend shows "Local: localhost:3000"
- [ ] Browser opens localhost:3000 without error
- [ ] Dashboard login page appears
- [ ] Can login successfully
- [ ] Camera feed loads
- [ ] No CORS errors in browser console

✅ **All checked? You're done!** System ready to monitor!

---

## 📞 NEED HELP?

Check these files:
- 🔧 Technical details → `DISTRIBUTED_SETUP.md`
- 👥 Friend guide → `FRIEND_SETUP.md`
- ✓ Full checklist → `COMPLETE_CHECKLIST.md`
- 🔐 Environment vars → `ENV_TEMPLATES.md`

