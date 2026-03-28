"""
Women Safety AI — Non-Blocking Alert Dispatcher
=================================================
Enqueues alert jobs to Redis/RQ so the inference thread is never
blocked by SMS sending or database writes.

Run the worker process::

    python -m alerts.dispatcher
"""

import os
import time
from datetime import datetime

import cv2
import redis as redis_lib
import rq

from config.settings import settings

_redis_conn = None
_alert_queue = None


def _get_queue() -> rq.Queue:
    """Lazy-init Redis connection and RQ queue named 'alerts'."""
    global _redis_conn, _alert_queue
    if _alert_queue is None:
        _redis_conn = redis_lib.Redis.from_url(settings.redis_url)
        _alert_queue = rq.Queue("alerts", connection=_redis_conn)
    return _alert_queue


def dispatch_alert(alert_payload: dict):
    """
    Public interface called from ``PipelineEngine._dispatch_alert()``.

    1. Saves snapshot frame to disk (if present).
    2. Strips the non-serializable ``frame`` numpy array.
    3. Enqueues ``_process_alert_job`` on the ``alerts`` queue.
    """
    payload = dict(alert_payload)  # shallow copy

    # Save snapshot to disk if frame is present
    frame = payload.pop("frame", None)
    if frame is not None:
        os.makedirs(settings.snapshot_dir, exist_ok=True)
        ts = int(payload.get("timestamp", time.time()))
        snapshot_filename = f"{ts}.jpg"
        snapshot_path = os.path.join(settings.snapshot_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
        payload["snapshot_path"] = snapshot_path

    try:
        _get_queue().enqueue(
            _process_alert_job,
            payload,
            job_timeout=30,
        )
    except (redis_lib.exceptions.ConnectionError, ConnectionRefusedError):
        print("  [WARN] Redis is not running. Processing alert synchronously instead of using RQ.")
        _process_alert_job(payload)


def _process_alert_job(payload: dict):
    """
    Runs inside an RQ worker process.

    1. Opens a DB session.
    2. Creates an ``Incident`` row with all model scores.
    3. Calls ``send_sms_alert()``.
    4. Creates an ``Alert`` row recording SMS result.
    5. Commits the transaction.
    """
    from db.session import SessionLocal
    from db.models import Incident, Alert
    from alerts.sms import send_sms_alert

    db = SessionLocal()
    try:
        incident = Incident(
            camera_id=payload.get("camera_id"),
            track_id=payload.get("woman_track_id"),
            incident_type=payload.get("incident_type", "unknown"),
            confidence=payload.get("fusion_score", 0.0),
            fusion_score=payload.get("fusion_score"),
            videomae_score=payload.get("videomae_score"),
            bilstm_score=payload.get("bilstm_score"),
            optflow_score=payload.get("optflow_score"),
            clip_score=payload.get("clip_score"),
            pose_score=payload.get("pose_score"),
            snapshot_path=payload.get("snapshot_path"),
        )
        db.add(incident)
        db.flush()  # get incident.id

        # Send SMS
        camera_name = f"Camera {payload.get('camera_id', '?')}"
        sms_result = send_sms_alert(
            camera_name=camera_name,
            incident_type=payload.get("incident_type", "unknown"),
            confidence=payload.get("fusion_score", 0.0),
            timestamp=payload.get("timestamp", time.time()),
        )

        # Record alert
        alert = Alert(
            incident_id=incident.id,
            channel="sms",
            status=sms_result["status"],
            error_msg=sms_result.get("error"),
            sent_at=datetime.utcnow() if sms_result["status"] == "sent" else None,
        )
        db.add(alert)
        db.commit()

        # Broadcast to WebSocket clients (best-effort)
        try:
            from api.websocket.alerts import broadcast_alert_sync
            broadcast_alert_sync({
                "camera_id": payload.get("camera_id"),
                "incident_type": payload.get("incident_type"),
                "fusion_score": payload.get("fusion_score"),
                "incident_id": incident.id,
                "timestamp": payload.get("timestamp"),
            })
        except Exception:
            pass  # WebSocket broadcast is best-effort

        return incident.id

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def start_worker():
    """Entry point for the RQ worker process."""
    conn = redis_lib.Redis.from_url(settings.redis_url)
    worker = rq.Worker(["alerts"], connection=conn)
    worker.work()


if __name__ == "__main__":
    start_worker()
