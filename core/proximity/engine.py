"""
Women Safety AI — Proximity Engine
=====================================
Tracks female isolation time and emits `proximity_violation` events
when a confirmed lone woman is approached by a male.

Design
------
- Maintains a per-track isolation timer (_isolation_start dict).
- Mutates TrackedPerson objects in-place (is_lone_woman, isolation_seconds).
- Returns a list of event dicts on each call — empty when nothing notable.
- An event is only fired when `was_lone == True` *before* the male entered
  the radius; random male proximity without prior isolation is ignored.

Usage:
    engine = ProximityEngine(radius_px=150, isolation_threshold=5.0)
    events = engine.update(persons)   # List[TrackedPerson]
    for e in events:
        print(e["type"], e["woman_track_id"], e["male_track_id"])
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

# Avoid circular import — TrackedPerson is defined in tracking.tracker
from core.tracking.tracker import TrackedPerson


class ProximityEngine:
    """
    Lone-woman detection and proximity violation emitter.

    Args:
        radius_px:           Distance threshold (pixels) — male is "nearby"
                             when within this Euclidean distance of the woman.
        isolation_threshold: Seconds a woman must be alone before she is
                             classified as a lone woman.
    """

    def __init__(
        self,
        radius_px: int = 150,
        isolation_threshold: float = 5.0,
    ) -> None:
        self.radius = int(radius_px)
        self.isolation_threshold = float(isolation_threshold)

        # Maps track_id → Unix timestamp when isolation *started*
        self._isolation_start: Dict[int, float] = {}

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _centre(bbox: List[float]) -> np.ndarray:
        """Return the 2-D centre of a bounding box [x1, y1, x2, y2]."""
        return np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float64,
        )

    def _dist(self, b1: List[float], b2: List[float]) -> float:
        """Euclidean distance between the centres of two bounding boxes."""
        return float(np.linalg.norm(self._centre(b1) - self._centre(b2)))

    # ── Main update ───────────────────────────────────────────────────────

    def update(self, persons: List[TrackedPerson]) -> List[dict]:
        """
        Process one frame's worth of tracked persons.

        Mutates each TrackedPerson in-place:
          - ``is_lone_woman``     → True once isolated ≥ isolation_threshold s
          - ``isolation_seconds`` → running timer (0 when male is nearby)

        Returns:
            List of proximity_violation event dicts.  Empty when nothing
            notable happened.  Each event has:

            .. code-block:: python

                {
                    "type":           "proximity_violation",
                    "woman_track_id": int,
                    "male_track_id":  int,
                    "woman_bbox":     List[float],
                    "male_bbox":      List[float],
                    "distance_px":    float,
                    "timestamp":      float,   # Unix time
                }
        """
        now = time.time()

        # Partition by gender
        women = [p for p in persons if p.gender == "female"]
        males  = [p for p in persons if p.gender == "male"]

        events: List[dict] = []

        for woman in women:
            # Find males within the radius
            nearby_males = [
                m for m in males
                if self._dist(woman.bbox, m.bbox) <= self.radius
            ]

            if not nearby_males:
                # ── No male nearby → run isolation timer ─────────────────
                if woman.track_id not in self._isolation_start:
                    self._isolation_start[woman.track_id] = now

                woman.isolation_seconds = now - self._isolation_start[woman.track_id]
                woman.is_lone_woman = (
                    woman.isolation_seconds >= self.isolation_threshold
                )

            else:
                # ── Male nearby → reset isolation, possibly emit event ────
                was_lone = woman.is_lone_woman          # snapshot before reset

                # Reset timer (key may not exist if woman was never alone)
                self._isolation_start.pop(woman.track_id, None)
                woman.isolation_seconds = 0.0
                woman.is_lone_woman = False

                if was_lone:
                    # Confirmed lone woman just had a male enter her radius
                    for male in nearby_males:
                        events.append({
                            "type":           "proximity_violation",
                            "woman_track_id": woman.track_id,
                            "male_track_id":  male.track_id,
                            "woman_bbox":     woman.bbox,
                            "male_bbox":      male.bbox,
                            "distance_px":    self._dist(woman.bbox, male.bbox),
                            "timestamp":      now,
                        })

        # ── Cleanup stale isolation entries ──────────────────────────────
        current_ids = {p.track_id for p in persons}
        stale = [tid for tid in self._isolation_start if tid not in current_ids]
        for tid in stale:
            del self._isolation_start[tid]

        return events
