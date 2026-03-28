"""
Women Safety AI — Weighted Fusion Layer
=========================================
Combines 5 detection signals into a single weighted score and decides
whether to trigger an alert.

Signals:
  1. videomae_score  — from ViolenceDetector   (weight=0.30)
  2. bilstm_score    — from AssaultDetector     (weight=0.25)
  3. optflow_score   — from AnomalyDetector     (weight=0.20)
  4. clip_score      — from CLIPContext         (weight=0.15)
  5. pose_score      — from PoseEstimator       (weight=0.10)

Usage:
    fusion = FusionLayer()
    inp = FusionInput(videomae_score=0.9, bilstm_score=0.8, ...)
    result = fusion.compute(inp)
    if result.triggered:
        dispatch_alert(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionInput:
    """Input scores from all 5 detection signals."""

    videomae_score: float = 0.0   # from ViolenceDetector
    bilstm_score: float = 0.0     # from AssaultDetector
    optflow_score: float = 0.0    # from AnomalyDetector
    clip_score: float = 0.0       # from CLIPContext
    pose_score: float = 0.0       # from PoseEstimator distress_score


@dataclass
class FusionResult:
    """Output of the fusion layer."""

    fusion_score: float          # final weighted score
    triggered: bool              # True if score >= threshold
    input_scores: FusionInput    # preserved for logging


class FusionLayer:
    """Weighted fusion of 5 signals with configurable threshold and weights."""

    DEFAULT_WEIGHTS = {
        "videomae": 0.30,
        "bilstm":   0.25,
        "optflow":  0.20,
        "clip":     0.15,
        "pose":     0.10,
    }

    def __init__(
        self,
        threshold: float = 0.75,
        weights: dict = None,
    ) -> None:
        self.threshold = threshold
        self.weights = dict(weights) if weights else dict(self.DEFAULT_WEIGHTS)

    # ── Compute fusion ────────────────────────────────────────────────────────

    def compute(self, inp: FusionInput) -> FusionResult:
        """
        Compute the weighted fusion score from all 5 signals.

        Formula:
            score = w_videomae * videomae + w_bilstm * bilstm
                  + w_optflow * optflow + w_clip * clip + w_pose * pose

        Returns:
            FusionResult with score rounded to 4 decimal places.
        """
        score = (
            self.weights["videomae"] * inp.videomae_score
            + self.weights["bilstm"]   * inp.bilstm_score
            + self.weights["optflow"]  * inp.optflow_score
            + self.weights["clip"]     * inp.clip_score
            + self.weights["pose"]     * inp.pose_score
        )
        score = round(score, 4)
        triggered = score >= self.threshold

        return FusionResult(
            fusion_score=score,
            triggered=triggered,
            input_scores=inp,
        )

    # ── Dynamic reconfiguration ───────────────────────────────────────────────

    def update_weights(self, w: dict) -> None:
        """
        Update fusion weights. Must sum to approximately 1.0.

        Raises:
            AssertionError: If weights sum deviates from 1.0 by > 0.01.
        """
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, (
            f"Fusion weights must sum to 1.0 (got {total:.4f}). "
            f"Provided weights: {w}"
        )
        self.weights = dict(w)

    def update_threshold(self, t: float) -> None:
        """
        Update the alert trigger threshold.

        Raises:
            AssertionError: If threshold is not in (0.0, 1.0).
        """
        assert 0.0 < t < 1.0, (
            f"Threshold must be between 0.0 and 1.0 exclusive (got {t})"
        )
        self.threshold = t
