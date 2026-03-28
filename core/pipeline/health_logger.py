"""
Women Safety AI — System Health Background Logger
====================================================
Writes VRAM usage, CPU usage, and FPS to the `system_health`
table every 10 seconds. Runs as a daemon thread so it never
blocks the inference pipeline.

Usage:
    from core.pipeline.health_logger import HealthLogger
    health_logger = HealthLogger(engines=ENGINES)
    health_logger.start()
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict

import psutil
import torch

if TYPE_CHECKING:
    from core.pipeline.engine import PipelineEngine


class HealthLogger:
    """Background health logger — writes VRAM, CPU, and FPS to ``system_health``."""

    def __init__(
        self,
        engines: Dict[int, "PipelineEngine"],
        interval_seconds: float = 10.0,
    ) -> None:
        self.engines = engines
        self.interval_seconds = interval_seconds
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = False

    def start(self) -> None:
        """Start the background health-logging thread."""
        self._running = True
        self._thread.start()
        print("[HealthLogger] Started — logging every "
              f"{self.interval_seconds}s")

    def stop(self) -> None:
        """Stop the health-logging thread."""
        self._running = False

    def _loop(self) -> None:
        """Internal loop — runs forever in a daemon thread."""
        while self._running:
            time.sleep(self.interval_seconds)
            try:
                self._log_snapshot()
            except Exception as exc:
                print(f"[HealthLogger] Error: {exc}")

    def _log_snapshot(self) -> None:
        """Take one health snapshot and write to database."""
        from db.session import SessionLocal
        from db.models import SystemHealth

        session = SessionLocal()
        try:
            for camera_id, engine in self.engines.items():
                vram_used = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                )
                cpu_pct = psutil.cpu_percent(interval=None)
                fps = engine.stream.fps if hasattr(engine, "stream") else 0

                row = SystemHealth(
                    camera_id=camera_id,
                    fps=fps,
                    vram_used=vram_used,
                    cpu_percent=cpu_pct,
                    logged_at=datetime.utcnow(),
                )
                session.add(row)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
