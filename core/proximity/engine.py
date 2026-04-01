"""
Women Safety AI — Proximity & Situation Engine
=================================================
Tracks female isolation time, detects proximity violations,
"woman surrounded by men" scenarios, and low-light/night conditions.

Events emitted:
  • proximity_violation  — lone woman approached by male
  • woman_surrounded     — female has 3+ males within extended radius
  • lone_woman_night     — lone woman detected in low-light conditions

Usage:
    engine = ProximityEngine(radius_px=150, isolation_threshold=5.0)
    events = engine.update(persons, frame)
    for e in events:
        print(e["type"], e["woman_track_id"])
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.tracking.tracker import TrackedPerson


class ProximityEngine:
    """
    Lone-woman detection, proximity violation, surrounded detection,
    and night/low-light detection.
    """

    SURROUND_COUNT = 3         # males needed for "surrounded" event
    BRIGHTNESS_THRESHOLD = 60  # mean brightness below this = night mode

    def __init__(
        self,
        radius_px: int = 150,
        isolation_threshold: float = 5.0,
    ) -> None:
        self.radius = int(radius_px)
        self.isolation_threshold = float(isolation_threshold)

        # Maps track_id → Unix timestamp when isolation *started*
        self._isolation_start: Dict[int, float] = {}

        # Track which surrounded events were already fired (to avoid spam)
        self._surrounded_fired: Dict[int, float] = {}

        # Night mode state
        self.is_night: bool = False

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

    @staticmethod
    def detect_night(frame: np.ndarray) -> bool:
        """
        Detect low-light / night conditions from frame brightness.

        Uses the V channel of HSV colour space. Works with both
        regular cameras (dark scenes) and IR cameras (low overall brightness
        even though image appears grey/green).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_brightness = float(hsv[:, :, 2].mean())
        return mean_brightness < ProximityEngine.BRIGHTNESS_THRESHOLD

    # ── Main update ───────────────────────────────────────────────────────

    def update(
        self,
        persons: List[TrackedPerson],
        frame: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Process one frame's worth of tracked persons.

        Args:
            persons: List of TrackedPerson objects with gender set.
            frame:   Optional BGR frame for night/brightness detection.

        Mutates each TrackedPerson in-place:
          - ``is_lone_woman``     → True once isolated ≥ isolation_threshold s
          - ``isolation_seconds`` → running timer

        Returns:
            List of event dicts (proximity_violation, woman_surrounded,
            lone_woman_night). Empty when nothing notable.
        """
        now = time.time()

        # ── Night detection ─────────────────────────────────────────────
        if frame is not None:
            self.is_night = self.detect_night(frame)

        # Partition by gender
        women = [p for p in persons if p.gender == "female"]
        males = [p for p in persons if p.gender == "male"]

        events: List[dict] = []

        for woman in women:
            # Find males within the radius
            nearby_males = [
                m for m in males
                if self._dist(woman.bbox, m.bbox) <= self.radius
            ]

            # Find males within extended radius (1.5x) for "surrounded"
            extended_males = [
                m for m in males
                if self._dist(woman.bbox, m.bbox) <= self.radius * 1.5
            ]

            if not nearby_males:
                # ── No male nearby → run isolation timer ─────────────────
                if woman.track_id not in self._isolation_start:
                    self._isolation_start[woman.track_id] = now

                woman.isolation_seconds = now - self._isolation_start[woman.track_id]
                woman.is_lone_woman = (
                    woman.isolation_seconds >= self.isolation_threshold
                )

                # ── Lone woman at night ──────────────────────────────────
                if woman.is_lone_woman and self.is_night:
                    events.append({
                        "type":           "lone_woman_night",
                        "woman_track_id": woman.track_id,
                        "woman_bbox":     woman.bbox,
                        "isolation_secs": woman.isolation_seconds,
                        "timestamp":      now,
                    })

            else:
                # ── Male nearby → reset isolation, possibly emit event ────
                was_lone = woman.is_lone_woman

                self._isolation_start.pop(woman.track_id, None)
                woman.isolation_seconds = 0.0
                woman.is_lone_woman = False

                if was_lone:
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

            # ── Woman surrounded detection ────────────────────────────────
            if len(extended_males) >= self.SURROUND_COUNT:
                last_fired = self._surrounded_fired.get(woman.track_id, 0)
                if (now - last_fired) > 10.0:  # cooldown: 10 seconds
                    self._surrounded_fired[woman.track_id] = now
                    events.append({
                        "type":           "woman_surrounded",
                        "woman_track_id": woman.track_id,
                        "woman_bbox":     woman.bbox,
                        "male_count":     len(extended_males),
                        "male_track_ids": [m.track_id for m in extended_males],
                        "timestamp":      now,
                    })

        # ── Cleanup stale isolation entries ──────────────────────────────
        current_ids = {p.track_id for p in persons}
        stale = [tid for tid in self._isolation_start if tid not in current_ids]
        for tid in stale:
            del self._isolation_start[tid]
        stale_surr = [tid for tid in self._surrounded_fired if tid not in current_ids]
        for tid in stale_surr:
            del self._surrounded_fired[tid]

        return events

    # ── Query methods ─────────────────────────────────────────────────────

    def get_lone_woman_ids(self, persons: List[TrackedPerson]) -> set:
        """Return set of track_ids currently flagged as lone women."""
        return {p.track_id for p in persons if p.is_lone_woman}

    def get_surrounded_ids(self, persons: List[TrackedPerson]) -> set:
        """Return track_ids of women currently surrounded."""
        women = [p for p in persons if p.gender == "female"]
        males = [p for p in persons if p.gender == "male"]
        result = set()
        for woman in women:
            nearby = sum(
                1 for m in males
                if self._dist(woman.bbox, m.bbox) <= self.radius * 1.5
            )
            if nearby >= self.SURROUND_COUNT:
                result.add(woman.track_id)
        return result
