"""
Women Safety AI — RAFT Optical Flow Anomaly Detector
=====================================================
Uses RAFT (Recurrent All-Pairs Field Transforms) to compute dense
optical flow between consecutive frames. Scene motion magnitude is
compared to an adaptive baseline to score anomalous movement.

Calibration Phase:
  First CALIBRATION_FRAMES (300) frames establish the normal motion
  baseline (mean + std of flow magnitudes). During this phase,
  anomaly_score is always 0.0.

Scoring Phase:
  After calibration, z-score of current flow magnitude relative to
  baseline is computed. Score ramps from 0 at 2σ to 1.0 at 5σ.

Usage:
    ad = AnomalyDetector()
    result = ad.compute(frame)
    print(result["anomaly_score"], result["is_anomaly"])
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class AnomalyDetector:
    """RAFT optical flow anomaly detector with adaptive baseline calibration."""

    CALIBRATION_FRAMES = 300  # ~20 seconds at 15 FPS

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Load RAFT small model
        print("[AnomalyDetector] Loading RAFT optical flow ...")
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.raft = self.raft.to(self.device).eval()
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            self.raft = torch.compile(self.raft, mode="reduce-overhead")
            print(f"[{self.__class__.__name__}] torch.compile applied")
        print("[AnomalyDetector] ✓ Loaded")

        # State
        self._prev: Optional[torch.Tensor] = None
        self._calib_buf: list = []
        self._calibrated: bool = False
        self._mean: float = 0.0
        self._std: float = 1.0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert BGR frame → RGB → torch tensor (1, 3, H, W) float32.
        Resize to (256, 256) — RAFT requires even spatial dimensions.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256))
        # HWC → CHW → add batch dim → float32 [0, 255]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    # ── Main API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(self, frame: np.ndarray) -> dict:
        """
        Compute optical flow anomaly score for the current frame.

        Args:
            frame: BGR uint8 numpy array.

        Returns:
            dict with keys:
              flow_magnitude (float),
              anomaly_score (float 0–1),
              is_anomaly (bool)
        """
        curr = self._to_tensor(frame)

        # First frame — no previous frame to compare
        if self._prev is None:
            self._prev = curr
            return {
                "flow_magnitude": 0.0,
                "anomaly_score": 0.0,
                "is_anomaly": False,
            }

        # Compute RAFT flow (take last flow estimate from the list)
        flow_list = self.raft(self._prev, curr)
        flow = flow_list[-1]   # (1, 2, H, W)

        # Compute magnitude
        flow_magnitude = float(torch.norm(flow, dim=1).mean().item())

        # Update previous frame
        self._prev = curr

        # ── Calibration phase ─────────────────────────────────────────────
        if not self._calibrated:
            self._calib_buf.append(flow_magnitude)
            if len(self._calib_buf) >= self.CALIBRATION_FRAMES:
                self._mean = float(np.mean(self._calib_buf))
                self._std = float(np.std(self._calib_buf)) + 1e-6
                self._calibrated = True
            return {
                "flow_magnitude": flow_magnitude,
                "anomaly_score": 0.0,
                "is_anomaly": False,
            }

        # ── Scoring phase ─────────────────────────────────────────────────
        z = (flow_magnitude - self._mean) / self._std
        anomaly_score = float(np.clip((z - 2.0) / 3.0, 0.0, 1.0))
        is_anomaly = anomaly_score >= 0.50

        return {
            "flow_magnitude": flow_magnitude,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }
