# 🚀 START HERE — How to Run Everything

**ngrok is MUCH SIMPLER than Cloudflare. Choose the guide based on your situation:**

---

## 🚀 RECOMMENDED: USE NGROK (SIMPLEST!)

👉 **Go to:** [NGROK_QUICKSTART.md](NGROK_QUICKSTART.md)

**Just 4 simple steps, takes 5 minutes total.**

```bash
1. choco install ngrok
2. Sign up at ngrok.com, copy token
3. ngrok config add-authtoken <token>
4. ngrok http 9000
5. Send your friend the URL
```

**This is the easiest way. Forget Cloudflare if you're just testing.**

---

## 📌 IF YOU WANT THE QUICKEST START (Cloudflare)

👉 **Go to:** [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md)

This is a 1-page cheat sheet with all the commands you need. Print it and keep it next to your screen.

**Only use this if you prefer Cloudflare** (it's more complex but permanent URL).

**Takes:** 5 minutes to read, ~3 minutes to run

---

---

## 🔧 IF YOU PREFER CLOUDFLARE (Advanced, but Permanent URL)

👉 **Go to:** [CLOUDFLARE_SETUP.md](CLOUDFLARE_SETUP.md)

More setup, but you get the same URL every time (good for production).

**Only if you need permanent URL or already have Cloudflare account.**

---

## 👀 IF YOU WANT TO SEE WHAT IT LOOKS LIKE

👉 **Go to:** [VISUAL_TERMINAL_GUIDE.md](VISUAL_TERMINAL_GUIDE.md)

Shows you exactly what should appear in each terminal with ASCII diagrams. Perfect for beginners.

**Takes:** 10 minutes to read + understand

---

## 📖 IF YOU WANT DETAILED STEP-BY-STEP

👉 **Go to:** [RUN_GUIDE.md](RUN_GUIDE.md)

Complete walkthrough with explanations, troubleshooting, and what to do if something goes wrong.

**Takes:** 20 minutes to read + understand

---

## ⚙️ IF YOU NEED TO SHARE WITH YOUR FRIEND

👉 **Send them:** [FRIEND_SETUP.md](FRIEND_SETUP.md)

This is written specifically for the person running the backend and frontend.

---

## 🔐 IF YOU'RE STUCK ON ENVIRONMENT VARIABLES

👉 **Go to:** [ENV_TEMPLATES.md](ENV_TEMPLATES.md)

Copy-paste templates for `.env` files. Shows exactly what to put in.

---

## ✓ IF YOU NEED A COMPLETE CHECKLIST

👉 **Go to:** [COMPLETE_CHECKLIST.md](COMPLETE_CHECKLIST.md)

Full checklist for both you and your friend. Good to verify everything is set up correctly.

---

## 🏗️ IF YOU WANT TO UNDERSTAND THE ARCHITECTURE

👉 **Go to:** [DISTRIBUTED_SETUP.md](DISTRIBUTED_SETUP.md)

Technical deep-dive into how the system works. For understanding, not for running.

---

---

## 🎯 RECOMMENDED READING ORDER

### First Time Setup:

1. ✅ [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md) — Get familiar
2. ✅ [VISUAL_TERMINAL_GUIDE.md](VISUAL_TERMINAL_GUIDE.md) — See what's normal
3. ✅ [ENV_TEMPLATES.md](ENV_TEMPLATES.md) — Set up .env files
4. ✅ [RUN_GUIDE.md](RUN_GUIDE.md) — Run it
5. ✅ [COMPLETE_CHECKLIST.md](COMPLETE_CHECKLIST.md) — Verify everything works

### Daily Use:

- [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md) — Just copy commands

### Troubleshooting:

- [RUN_GUIDE.md](RUN_GUIDE.md) — Look for your error in "Common Issues"
- [COMPLETE_CHECKLIST.md](COMPLETE_CHECKLIST.md) — Run through checklist again

---

## ⚡ 60-SECOND SUMMARY

**What you need to do:**

1. **Your laptop (GPU machine):**
   - Run: `cloudflared tunnel run women-safety-model` (Terminal 1)
   - Run: `python model_server.py` (Terminal 2)
   - Keep both running
   - Get your model URL: `https://model.yourdomain.com`

2. **Friend's laptop:**
   - Set `.env`: `MODEL_API_URL=https://model.yourdomain.com` (your URL from above)
   - Run: `python -m uvicorn api.main:app --port 8000` (Terminal 1)
   - Run: `npm run dev` in `dashboard/` folder (Terminal 2)
   - Open: `http://localhost:3000` in browser

3. **Access dashboard:**
   - URL: `http://localhost:3000`
   - Login and monitor!

**Done!** 🎉

---

## 🆘 IF SOMETHING'S NOT WORKING

### The 3 Steps to Debug:

1. **Check all 4 terminals are running:**
   ```
   ✅ Your laptop T1: Cloudflare tunnel
   ✅ Your laptop T2: Model server
   ✅ Friend's laptop T1: Backend
   ✅ Friend's laptop T2: Frontend
   ```

2. **Check .env files:**
   ```
   ✅ Your .env: USE_REMOTE_MODEL=false
   ✅ Friend's .env: USE_REMOTE_MODEL=true
   ✅ Friend's .env: MODEL_API_URL=https://model.yourdomain.com
   ```

3. **Check URLs work:**
   ```bash
   # From friend's laptop:
   curl https://model.yourdomain.com/
   # Should return: {"service":"Women Safety AI..."}
   ```

If still stuck → Read [RUN_GUIDE.md](RUN_GUIDE.md#common-issues-while-running)

---

## 📋 FILES YOU HAVE

| File | Purpose | Read When |
|------|---------|-----------|
| [NGROK_QUICKSTART.md](NGROK_QUICKSTART.md) | **⭐ RECOMMENDED: ngrok setup** | **First! Super simple** |
| [NGROK_SETUP.md](NGROK_SETUP.md) | Detailed ngrok guide | Want more details |
| [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md) | Command cheat sheet (Cloudflare) | Running daily |
| [VISUAL_TERMINAL_GUIDE.md](VISUAL_TERMINAL_GUIDE.md) | What should appear | First time setup |
| [RUN_GUIDE.md](RUN_GUIDE.md) | Complete instruction manual | Need detailed steps |
| [CLOUDFLARE_SETUP.md](CLOUDFLARE_SETUP.md) | Cloudflare guide (advanced) | Only if using Cloudflare |
| [FRIEND_SETUP.md](FRIEND_SETUP.md) | Guide for your friend | Send to friend |
| [ENV_TEMPLATES.md](ENV_TEMPLATES.md) | Environment variables | Setting up .env |
| [COMPLETE_CHECKLIST.md](COMPLETE_CHECKLIST.md) | Full verification checklist | After setup to verify |
| [DISTRIBUTED_SETUP.md](DISTRIBUTED_SETUP.md) | Architecture explanation | Understanding how it works |
| [RUN_GUIDE.md](RUN_GUIDE.md#quick-fixes) | Quick fixes table | Troubleshooting |

---

## 🎬 VIDEO WALKTHROUGH (What You'd See)

If this was a video, it would be:

```
[0:00] Open 4 terminals
[0:15] Terminal 1: cloudflared tunnel...
       (wait 3s) "tunnel running" appears ✓
       
[0:20] Terminal 2: python model_server.py
       (wait 120s) "models loaded" appears ✓
       
[2:25] Terminal 3 (friend): python -m uvicorn...
       (wait 10s) "Uvicorn running" appears ✓
       
[2:35] Terminal 4 (friend): npm run dev
       (wait 10s) "Ready" appears ✓
       
[2:50] Browser: http://localhost:3000
       Dashboard loads ✓
       
[3:00] Login → See camera feed → System monitoring ✓
```

Total time: **3 minutes** (after models are cached)

---

## ✅ SUCCESS CHECKLIST

Before you're done:

- [ ] Cloudflare tunnel is running on your machine
- [ ] Model server is running on your machine
- [ ] Model server shows "all models loaded"
- [ ] Friend's backend is running
- [ ] Friend's frontend is running
- [ ] Browser shows `http://localhost:3000`
- [ ] Login page appears
- [ ] No CORS errors in browser console (F12)
- [ ] After login, dashboard loads
- [ ] Camera feed is visible
- [ ] Detection boxes appear on video

✅ **All checked? You're done!**

---

## 🔁 DAILY ROUTINE

**Morning:**
```bash
# Terminal 1 (your machine)
cloudflared tunnel run women-safety-model

# Terminal 2 (your machine)
python model_server.py

# Terminal 1 (friend's machine)
python -m uvicorn api.main:app --port 8000

# Terminal 2 (friend's machine)
cd dashboard && npm run dev

# Browser (friend's machine)
http://localhost:3000
```

**Evening:**
```bash
# Close all 4 terminals
Ctrl+C in each
```

---

## 💡 TIPS

- 💾 **Save this file as a browser bookmark** — You'll reference it often
- 📋 **Print [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md)** — Keep on desk
- 🔔 **Create a checklist** — Copy the one from [COMPLETE_CHECKLIST.md](COMPLETE_CHECKLIST.md)
- 📱 **Add phone reminder** — Set alarm "Start model server"
- 📧 **Share files with friend** — Email them [FRIEND_SETUP.md](FRIEND_SETUP.md)

---

## 🎓 LEARNING PATH

Want to understand the system better?

1. **"What's happening?"** → Read intro of [DISTRIBUTED_SETUP.md](DISTRIBUTED_SETUP.md)
2. **"How do I run it?"** → Read [VISUAL_TERMINAL_GUIDE.md](VISUAL_TERMINAL_GUIDE.md)
3. **"What if it breaks?"** → Read [RUN_GUIDE.md](RUN_GUIDE.md#common-issues-while-running)
4. **"Deep dive"** → Read all of [DISTRIBUTED_SETUP.md](DISTRIBUTED_SETUP.md)

---

## 🚀 READY?

Pick an option above and start!

👉 **Recommended:** Start with [QUICK_TERMINAL_GUIDE.md](QUICK_TERMINAL_GUIDE.md)

---

---

**Questions? Check [RUN_GUIDE.md](RUN_GUIDE.md) — it has answers to most issues!**

