"""
Women Safety AI — BoT-SORT Person Tracker
===========================================
Wraps Ultralytics YOLO + BoT-SORT to assign persistent track_ids.
Maintains a state dictionary {track_id: TrackedPerson} across frames.
Velocity is computed from bbox centre displacement.
Stale tracks (not seen for 2 s) are pruned.

Usage:
    tracker = PersonTracker()
    persons = tracker.update(frame)
    for p in persons:
        print(p.track_id, p.bbox, p.gender)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class TrackedPerson:
    """Per-person tracking state maintained across frames."""

    track_id: int
    bbox: List[float]                              # [x1, y1, x2, y2]
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [dx, dy] px/frame
    gender: str = "unknown"                        # male | female | unknown
    gender_conf: float = 0.0
    is_lone_woman: bool = False                    # set by ProximityEngine (Sprint 2)
    isolation_seconds: float = 0.0                 # set by ProximityEngine (Sprint 2)
    frames_tracked: int = 0
    last_seen: float = field(default_factory=time.time)


class PersonTracker:
    """BoT-SORT multi-object tracker with persistent state management."""

    STALE_TIMEOUT = 2.0  # seconds before pruning a lost track

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Load YOLO model (used for combined detection + tracking)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self._half = self.device == "cuda"

        # Persistent state: {track_id: TrackedPerson}
        self._state: Dict[int, TrackedPerson] = {}

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _centre(bbox: List[float]) -> tuple:
        """Return (cx, cy) of a bbox [x1, y1, x2, y2]."""
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    # ── main update ─────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Run detection + tracking on a frame.

        Returns:
            List of currently active TrackedPerson objects.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],       # person only
            conf=0.5,
            verbose=False,
            half=self._half,
        )

        now = time.time()
        active_ids: set = set()

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox_arr, tid in zip(boxes, ids):
                bbox = bbox_arr.tolist()
                active_ids.add(tid)

                if tid in self._state:
                    # Existing track — compute velocity & update
                    old = self._state[tid]
                    cx_old, cy_old = self._centre(old.bbox)
                    cx_new, cy_new = self._centre(bbox)
                    old.velocity = [cx_new - cx_old, cy_new - cy_old]
                    old.bbox = bbox
                    old.frames_tracked += 1
                    old.last_seen = now
                else:
                    # New track
                    self._state[tid] = TrackedPerson(
                        track_id=tid,
                        bbox=bbox,
                        last_seen=now,
                    )

        # Prune stale tracks
        stale = [
            tid
            for tid, tp in self._state.items()
            if tid not in active_ids and (now - tp.last_seen) > self.STALE_TIMEOUT
        ]
        for tid in stale:
            del self._state[tid]

        # Return only currently active persons
        return [self._state[tid] for tid in active_ids if tid in self._state]

    @property
    def state(self) -> Dict[int, TrackedPerson]:
        """Read-only access to the full tracking state."""
        return self._state
