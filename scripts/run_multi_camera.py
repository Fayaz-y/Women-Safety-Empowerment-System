"""
Women Safety AI — Multi-Camera Launcher
=========================================
Starts one PipelineEngine per connected camera, sharing heavy GPU
models across all engines to stay within the 7.5 GB VRAM budget.

Usage:
    python scripts/run_multi_camera.py
"""

from __future__ import annotations

import signal
import sys
import time

import torch

# ── Camera configuration ─────────────────────────────────────────────────────
# Read-only for V1.  Future: load from DB via GET /api/v1/cameras.

CAMERAS = [
    {"camera_id": 1, "source": 0},
    # {"camera_id": 2, "source": 1}, # Commented out for users with only 1 webcam
]


def main():
    # Late imports so the worker can import this module without side effects
    from core.violence.detector import ViolenceDetector
    from core.assault.detector import AssaultDetector
    from core.anomaly.detector import AnomalyDetector
    from core.context.clip_context import CLIPContext
    from core.pipeline.engine import PipelineEngine
    from core.pipeline.health_logger import HealthLogger
    from alerts.dispatcher import dispatch_alert
    from api.websocket.stream import ENGINES

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load shared GPU-heavy models once ──────────────────────────────
    print("=" * 60)
    print("  Loading shared GPU models (loaded once, shared by all cameras)")
    print("=" * 60)

    shared_models = {}

    print("\n[Shared] ViolenceDetector ...")
    shared_models["violence"] = ViolenceDetector(device=device)

    print("[Shared] AssaultDetector ...")
    shared_models["assault"] = AssaultDetector(device=device)

    print("[Shared] AnomalyDetector ...")
    shared_models["anomaly"] = AnomalyDetector(device=device)

    print("[Shared] CLIPContext ...")
    shared_models["clip"] = CLIPContext(device=device)

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"\n  Shared models VRAM: {vram:.2f} GB\n")

    # ── 2. Alert callback ─────────────────────────────────────────────────
    def on_alert_callback(payload: dict):
        cam_id = payload.get("camera_id", "?")
        inc_type = payload.get("incident_type", "unknown")
        score = payload.get("fusion_score", 0.0)
        print(
            f"[ALERT] cam{cam_id} | {inc_type} | score={score:.3f}"
        )
        dispatch_alert(payload)

    # ── 3. Start one engine per camera ────────────────────────────────────
    engines: dict = {}

    for cfg in CAMERAS:
        cam_id = cfg["camera_id"]
        source = cfg["source"]

        print(f"\n--- Starting engine for Camera {cam_id} (source={source}) ---")

        engine = PipelineEngine(
            camera_id=cam_id,
            source=source,
            on_alert=on_alert_callback,
            device=device,
            shared_models=shared_models,
        )
        engine.load_models()
        engine.start()

        engines[cam_id] = engine
        ENGINES[cam_id] = engine          # register in WebSocket stream registry

        # Stagger startup to avoid VRAM spikes
        time.sleep(2)

    # ── 4. Start health logger ────────────────────────────────────────────
    health_logger = HealthLogger(engines=engines)
    health_logger.start()

    # ── 5. Signal handlers for clean shutdown ─────────────────────────────
    def _shutdown(sig, frame):
        print("\n[Shutdown] Stopping all engines ...")
        for eid, eng in engines.items():
            eng.stop()
            print(f"  cam{eid} stopped")
        health_logger.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 6. Main status loop ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All cameras running — press Ctrl+C to stop")
    print("=" * 60 + "\n")

    while True:
        for eid, eng in engines.items():
            fps = eng.stream.fps if hasattr(eng, "stream") else 0
            print(f"  cam{eid} FPS: {fps}")
        time.sleep(10)


if __name__ == "__main__":
    main()
