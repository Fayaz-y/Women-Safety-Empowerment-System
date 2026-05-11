"""
Women Safety AI — FastAPI Application Entry Point
====================================================
Creates the FastAPI app, registers CORS middleware,
mounts all REST routers and WebSocket routes.

On startup the app also launches camera pipeline engines so that
the WebSocket stream handler and the camera engines share the same
process (and therefore the same ``ENGINES`` dict).

Run::

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.auth import router as auth_router
from api.routes.cameras import router as cameras_router
from api.routes.incidents import router as incidents_router
from api.routes.system import router as system_router
from api.websocket.stream import endpoint as stream_endpoint, ENGINES
from api.websocket.alerts import endpoint as alerts_endpoint


# ── Camera bootstrap (runs in a background thread) ──────────────────────────

def _start_camera_engines():
    """Start one PipelineEngine per configured camera."""
    import time
    import torch
    import os
    import sys
    from config.settings import settings
    from core.pipeline.engine import PipelineEngine
    from alerts.dispatcher import dispatch_alert

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine camera source based on CAMERA_TOGGLE
    #   0 = laptop webcam (device index 0)
    #   1 = external USB cam (device index 1)
    #   2 = looped video file streaming (continuous via WebSocket)
    
    # Handle video file mode (toggle=2) — continuous streaming, not batch
    if settings.camera_toggle == 2:
        source = settings.camera_source_path
        if not source:
            print("[ERROR] CAMERA_TOGGLE=2 but CAMERA_SOURCE_PATH is empty!", flush=True)
            return
        
        # Verify file exists
        if not os.path.isfile(source):
            print(f"[ERROR] Video file not found: {source}", flush=True)
            return
        
        print(f"[STREAMING] Using looped video file: {source}", flush=True)
        print("[STREAMING] Video will loop continuously and stream via WebSocket\n", flush=True)
        # Source is now the file path for continuous looped streaming
    else:
        # Normal live camera streaming (toggle=0 or 1)
        source = settings.camera_toggle  # 0 or 1
        print(f"[STREAMING] Mode: device index {source}", flush=True)

    CAMERAS = [
        {"camera_id": 1, "source": source},
    ]

    def on_alert(payload: dict):
        cam_id = payload.get("camera_id", "?")
        inc_type = payload.get("incident_type", "unknown")
        score = payload.get("fusion_score", 0.0)
        print(f"[ALERT] cam{cam_id} | {inc_type} | score={score:.3f}")
        dispatch_alert(payload)

    for cfg in CAMERAS:
        cam_id = cfg["camera_id"]
        source = cfg["source"]
        print(f"\n--- Starting engine for Camera {cam_id} (source={source}) ---", flush=True)

        engine = PipelineEngine(
            camera_id=cam_id,
            source=source,
            on_alert=on_alert,
            device=device,
        )
        engine.load_models()
        engine.start()
        ENGINES[cam_id] = engine
        time.sleep(1)

    print("\n[API] All camera engines started and registered.\n", flush=True)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start camera engines in a daemon thread so uvicorn startup isn't blocked
    t = threading.Thread(target=_start_camera_engines, daemon=True)
    t.start()
    yield
    # Shutdown: stop all engines
    for eid, eng in ENGINES.items():
        eng.stop()
        print(f"  cam{eid} stopped")


app = FastAPI(
    title="Women Safety AI API",
    version="1.0.0",
    description="AI-powered real-time video surveillance for women safety",
    lifespan=lifespan,
)

# ── CORS ──
from config.settings import settings

# Parse allowed origins from settings
allowed_origins = [origin.strip() for origin in settings.allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST Routers ──
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(cameras_router, prefix="/api/v1/cameras", tags=["cameras"])
app.include_router(incidents_router, prefix="/api/v1/incidents", tags=["incidents"])
app.include_router(system_router, prefix="/api/v1/system", tags=["system"])

# ── WebSocket Routes ──
app.add_api_websocket_route("/ws/stream/{camera_id}", stream_endpoint)
app.add_api_websocket_route("/ws/alerts", alerts_endpoint)


@app.get("/")
def root():
    from api.websocket.stream import ENGINES
    return {
        "service": "Women Safety AI API",
        "version": "1.0.0",
        "engines_loaded": list(ENGINES.keys()),
        "engine_ready": 1 in ENGINES and ENGINES[1] is not None
    }
