"""
Women Safety AI — YOLO11-Pose Estimator + Distress Analyser
=============================================================
Extracts 17 COCO keypoints per tracked person using YOLO11-Pose.
Analyses body geometry to detect 4 distress flags and returns a
normalised distress_score in [0.0, 1.0].

COCO Keypoint Index Reference
------------------------------
 0  Nose          9  Left Wrist
 1  Left Eye     10  Right Wrist
 2  Right Eye    11  Left Hip
 3  Left Ear     12  Right Hip
 4  Right Ear    13  Left Knee
 5  Left Shld    14  Right Knee
 6  Right Shld   15  Left Ankle
 7  Left Elbow   16  Right Ankle
 8  Right Elbow

Each keypoint is [x, y, confidence]. Only used when conf >= 0.5.

Skeleton connection pairs (drawn by draw_skeleton):
  Head:  (0,1),(0,2),(1,3),(2,4)
  Arms:  (5,6),(5,7),(7,9),(6,8),(8,10)
  Torso: (5,11),(6,12),(11,12)
  Legs:  (11,13),(13,15),(12,14),(14,16)

Usage:
    pe = PoseEstimator()
    results = pe.estimate(frame)  # List[PoseResult]
    for r in results:
        frame = pe.draw_skeleton(frame, r)
        print(r.track_id, r.distress_score, r.flags)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


# ── Skeleton connection pairs ─────────────────────────────────────────────────
SKELETON_PAIRS: List[Tuple[int, int]] = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Minimum confidence to accept a keypoint
KP_CONF_THRESHOLD: float = 0.5


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PoseResult:
    """Per-person pose estimation result for one frame."""

    track_id: int
    """Persistent track_id matching TrackedPerson.track_id."""

    keypoints: np.ndarray
    """Shape (17, 3) — columns: [x, y, confidence] per COCO keypoint."""

    distress_score: float
    """Normalised distress level 0.0 (calm) → 1.0 (maximum distress)."""

    flags: Dict[str, bool] = field(default_factory=dict)
    """Boolean flags: raised_arms, covering_face, falling, hunched."""


# ── Main class ────────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    YOLO11-Pose pose estimator with distress analysis.

    Args:
        model_path: Ultralytics YOLO pose model file.
                    Defaults to 'yolo11n-pose.pt' (auto-downloaded).
        device:     'cuda' or 'cpu'. Falls back to CPU if CUDA unavailable.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str = "cuda",
    ) -> None:
        # ── Device selection ─────────────────────────────────────────────
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._half = self.device == "cuda"

        # ── Load YOLO11-Pose model ───────────────────────────────────────
        self.model = YOLO(model_path)
        self.model.to(self.device)

    # ── Public API ────────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Run YOLO11-Pose on a single frame.

        Calls model.track() with BoT-SORT so track_ids stay persistent
        across calls — the same IDs used by PersonTracker.

        Args:
            frame: BGR uint8 numpy array (any resolution).

        Returns:
            List of PoseResult objects, one per tracked person in the frame.
            Empty list if no persons are detected or all lack track IDs.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],        # person only
            verbose=False,
            half=self._half,
        )

        pose_results: List[PoseResult] = []

        for result in results:
            # Guard: need both boxes with IDs and keypoints
            if result.boxes is None or result.boxes.id is None:
                continue
            if result.keypoints is None:
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)
            # keypoints tensor shape: (N, 17, 3) — [x, y, conf]
            kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for i, tid in enumerate(ids):
                if i >= len(kp_data):
                    continue

                kp: np.ndarray = kp_data[i]          # (17, 3)
                flags, score = self._analyse_distress(kp)
                pose_results.append(
                    PoseResult(
                        track_id=int(tid),
                        keypoints=kp,
                        distress_score=score,
                        flags=flags,
                    )
                )

        return pose_results

    # ── Distress analysis ─────────────────────────────────────────────────

    def _analyse_distress(
        self, keypoints: np.ndarray
    ) -> Tuple[Dict[str, bool], float]:
        """
        Analyse body geometry from 17-keypoint array → (flags, score).

        Only keypoints with confidence >= 0.5 are used. Missing keypoints
        are treated as None so individual flags degrade gracefully.

        Args:
            keypoints: (17, 3) array — [x, y, conf] per COCO keypoint.

        Returns:
            (flags_dict, score_float)
            flags_dict keys: raised_arms, covering_face, falling, hunched
            score_float: float in [0.0, 1.0]
        """

        # ── Helper: extract keypoint or None ────────────────────────────
        def kp(idx: int) -> Optional[Tuple[float, float]]:
            """Return (x, y) if conf >= threshold, else None."""
            if idx >= len(keypoints):
                return None
            x, y, c = keypoints[idx]
            return (float(x), float(y)) if float(c) >= KP_CONF_THRESHOLD else None

        # Pre-extract all needed keypoints
        nose       = kp(0)
        l_shoulder = kp(5)
        r_shoulder = kp(6)
        l_wrist    = kp(9)
        r_wrist    = kp(10)
        l_hip      = kp(11)
        r_hip      = kp(12)
        l_ankle    = kp(15)
        r_ankle    = kp(16)

        # ── Flag: raised_arms ────────────────────────────────────────────
        # Wrist y < shoulder y means wrist is HIGHER in the image
        # (y increases downward; smaller y = higher position)
        raised_arms = False
        if l_wrist is not None and l_shoulder is not None:
            if l_wrist[1] < l_shoulder[1]:
                raised_arms = True
        if r_wrist is not None and r_shoulder is not None:
            if r_wrist[1] < r_shoulder[1]:
                raised_arms = True

        # ── Flag: covering_face ──────────────────────────────────────────
        # Either wrist within 60 px Euclidean distance from nose
        covering_face = False
        if nose is not None:
            if l_wrist is not None:
                dist = math.hypot(nose[0] - l_wrist[0], nose[1] - l_wrist[1])
                if dist < 60.0:
                    covering_face = True
            if r_wrist is not None:
                dist = math.hypot(nose[0] - r_wrist[0], nose[1] - r_wrist[1])
                if dist < 60.0:
                    covering_face = True

        # ── Flag: falling ────────────────────────────────────────────────
        # Body nearly horizontal: |avg_shoulder_y - avg_ankle_y| < 120
        falling = False
        if (l_shoulder is not None and r_shoulder is not None
                and l_ankle is not None and r_ankle is not None):
            sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            an_y = (l_ankle[1]   + r_ankle[1])   / 2.0
            if abs(sh_y - an_y) < 120.0:
                falling = True

        # ── Flag: hunched ────────────────────────────────────────────────
        # avg shoulder y > avg hip y + 30  →  shoulders appear BELOW hips
        # (heavy forward hunch pushes shoulders down in image coordinates)
        hunched = False
        if (l_shoulder is not None and r_shoulder is not None
                and l_hip is not None and r_hip is not None):
            avg_sh_y  = (l_shoulder[1] + r_shoulder[1]) / 2.0
            avg_hip_y = (l_hip[1]      + r_hip[1])      / 2.0
            if avg_sh_y > avg_hip_y + 30.0:
                hunched = True

        # ── Distress score ───────────────────────────────────────────────
        flags = {
            "raised_arms":   raised_arms,
            "covering_face": covering_face,
            "falling":       falling,
            "hunched":       hunched,
        }
        score = sum(flags.values()) / 4.0

        return flags, score

    # ── Skeleton drawing ──────────────────────────────────────────────────

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose: PoseResult,
    ) -> np.ndarray:
        """
        Draw the skeleton overlay onto the frame.

        Colour: red (0, 0, 255) if distress_score > 0.25, else green (0, 255, 128).
        Keypoints: yellow (255, 255, 0) filled circle radius 3.
        Connections: line between pairs where both endpoints have conf >= 0.5.

        Args:
            frame: BGR uint8 numpy array. Modified in-place.
            pose:  PoseResult for this person.

        Returns:
            The modified frame (same object as input).
        """
        import cv2  # import here so the module can be used without cv2 installed

        kp = pose.keypoints  # (17, 3)
        colour = (0, 0, 255) if pose.distress_score > 0.25 else (0, 255, 128)

        # Draw connection lines
        for idx_a, idx_b in SKELETON_PAIRS:
            if idx_a >= len(kp) or idx_b >= len(kp):
                continue
            xa, ya, ca = kp[idx_a]
            xb, yb, cb = kp[idx_b]
            if float(ca) >= KP_CONF_THRESHOLD and float(cb) >= KP_CONF_THRESHOLD:
                pt_a = (int(xa), int(ya))
                pt_b = (int(xb), int(yb))
                cv2.line(frame, pt_a, pt_b, colour, 2, cv2.LINE_AA)

        # Draw keypoint dots
        for x, y, c in kp:
            if float(c) >= KP_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1, cv2.LINE_AA)

        return frame
