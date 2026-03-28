"""
Women Safety AI — System Status & Config Routes
==================================================
GET /status  — live VRAM, FPS, uptime
GET /config  — all config key–values
PATCH /config — update config values
All endpoints require JWT auth.
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user
from db.models import SystemConfig, SystemHealth
from db.session import get_db

router = APIRouter()

_startup_time = time.time()


@router.get("/status")
def system_status(
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Return live system metrics."""
    uptime_seconds = round(time.time() - _startup_time, 1)

    # VRAM info (best-effort)
    vram_used = 0.0
    vram_total = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            vram_used = round(torch.cuda.memory_allocated() / 1e9, 2)
            vram_total = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    except Exception:
        pass

    # FPS per camera from latest health rows
    fps_per_camera = {}
    health_rows = (
        db.query(SystemHealth)
        .order_by(SystemHealth.logged_at.desc())
        .limit(20)
        .all()
    )
    seen = set()
    for row in health_rows:
        if row.camera_id not in seen:
            fps_per_camera[str(row.camera_id)] = row.fps
            seen.add(row.camera_id)

    return {
        "vram_used": vram_used,
        "vram_total": vram_total,
        "fps_per_camera": fps_per_camera,
        "uptime_seconds": uptime_seconds,
        "model_status": "loaded",
    }


@router.get("/config")
def get_config(
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Return all system_config rows as a flat dict."""
    rows = db.query(SystemConfig).all()
    return {row.key: row.value for row in rows}


@router.patch("/config")
def update_config(
    updates: dict,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Upsert config key–value pairs."""
    for key, value in updates.items():
        row = db.query(SystemConfig).filter(SystemConfig.key == key).first()
        if row:
            row.value = str(value)
            row.updated_at = datetime.utcnow()
        else:
            row = SystemConfig(key=key, value=str(value))
            db.add(row)
    db.commit()
    return {"ok": True, "updated": list(updates.keys())}
