"""
Women Safety AI — VideoMAE V2 Violence Detector
=================================================
Uses VideoMAE V2 pre-trained on Kinetics-400 to classify 16-frame clips
for violent actions. Maintains a per-track rolling buffer of 16 frames.
When the buffer is full, runs inference and maps the predicted action
label to a violence score.

Usage:
    vd = ViolenceDetector()
    vd.push_frame(track_id=7, crop=crop_bgr)
    result = vd.predict(track_id=7)
    if result and result["is_violent"]:
        print(f"Violence detected: {result['label']}")
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

# ── Violence label set (Kinetics-400 labels that map to violence) ────────────
VIOLENCE_CLASSES = {
    "wrestling",
    "punching_person_boxing",
    "punching_person",
    "slapping",
    "fighting",
    "pushing_person",
    "choking",
    "headbutting",
    "kicking_person",
    "grabbing",
}


class ViolenceDetector:
    """VideoMAE V2 violence detector with per-track 16-frame rolling buffer."""

    CLIP_LEN = 16               # frames required for one inference
    FRAME_SIZE = (224, 224)      # resize target for each frame
    CONFIDENCE_THRESHOLD = 0.80  # minimum confidence to classify as violent

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Load VideoMAE V2 processor + model
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        print(f"[ViolenceDetector] Loading {model_name} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device).half().eval()
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print(f"[{self.__class__.__name__}] torch.compile applied")
        print("[ViolenceDetector] ✓ Loaded")

        # Per-track rolling frame buffers: track_id → deque of (224,224,3) RGB arrays
        self._buffers: Dict[int, deque] = {}

    # ── Push frame ────────────────────────────────────────────────────────────

    def push_frame(self, track_id: int, crop: np.ndarray) -> None:
        """
        Push a BGR crop into the rolling buffer for this track.

        Args:
            track_id: Persistent track ID.
            crop:     BGR uint8 numpy array (person crop from frame).
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.CLIP_LEN)

        # Resize to (224, 224)
        resized = cv2.resize(crop, self.FRAME_SIZE)
        # Convert BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._buffers[track_id].append(rgb)

    # ── Predict ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, track_id: int) -> Optional[dict]:
        """
        Run VideoMAE V2 inference on the 16-frame buffer for this track.

        Returns:
            None if buffer doesn't exist or has < 16 frames.
            Otherwise dict with keys:
              track_id, label, confidence, is_violent, violence_score
        """
        if track_id not in self._buffers:
            return None
        if len(self._buffers[track_id]) < self.CLIP_LEN:
            return None

        # Collect the 16 frames from the deque
        frames = list(self._buffers[track_id])

        # Process through VideoMAE processor
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).half()

        # Forward pass
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

        # Get top prediction
        top_idx = int(torch.argmax(probs).item())
        conf = float(probs[top_idx].item())

        # Look up the label
        label = self.model.config.id2label[top_idx].lower().replace(" ", "_")

        # Check if label (or any substring) matches a violence class
        label_match = any(vc in label for vc in VIOLENCE_CLASSES)

        # Determine violence
        is_violent = label_match and conf >= self.CONFIDENCE_THRESHOLD

        # Compute violence score
        if is_violent:
            violence_score = conf
        else:
            violence_score = conf * 0.3  # small residual score

        return {
            "track_id": track_id,
            "label": label,
            "confidence": conf,
            "is_violent": is_violent,
            "violence_score": violence_score,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_track(self, track_id: int) -> None:
        """Remove track_id from buffers if present."""
        self._buffers.pop(track_id, None)
