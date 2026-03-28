"""
Women Safety AI — FastAPI Application Entry Point
====================================================
Creates the FastAPI app, registers CORS middleware,
mounts all REST routers and WebSocket routes.

Run::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload=False --workers=1
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.auth import router as auth_router
from api.routes.cameras import router as cameras_router
from api.routes.incidents import router as incidents_router
from api.routes.system import router as system_router
from api.websocket.stream import endpoint as stream_endpoint
from api.websocket.alerts import endpoint as alerts_endpoint

app = FastAPI(
    title="Women Safety AI API",
    version="1.0.0",
    description="AI-powered real-time video surveillance for women safety",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    return {"service": "Women Safety AI API", "version": "1.0.0"}
