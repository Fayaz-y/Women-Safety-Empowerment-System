"""
Women Safety AI — VGG19 + BiLSTM + Pose Fusion Assault Detector
================================================================
Custom architecture for sexual assault pattern detection combining:
  • VGG19 as spatial feature extractor (pretrained on ImageNet)
  • BiLSTM processing both visual features and pose keypoints over 16 frames

The model processes 16-frame sequences of person crops + keypoint data
to detect forced contact, resistance, and unwanted touch patterns.

Usage:
    ad = AssaultDetector()
    ad.push_frame(track_id=7, crop=crop_bgr, keypoints=kp_array)
    result = ad.predict(track_id=7)
    if result and result["is_assault"]:
        print(f"Assault detected: conf={result['assault_confidence']:.2f}")
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ── Private model architecture ────────────────────────────────────────────────

class _AssaultModel(nn.Module):
    """
    VGG19 spatial features + BiLSTM temporal model for assault detection.

    Architecture:
        1. VGG19 backbone (pretrained, classifier removed)
           → (B, 512, 7, 7) feature map → flatten → (B, VGG_FEAT_DIM)
        2. Visual projection: Linear(VGG_FEAT_DIM, 256) → ReLU
        3. Pose  projection: Linear(POSE_DIM, 64) → ReLU
        4. Concatenate: [256 + 64] = 320-dim per-frame feature
        5. BiLSTM(input_size=320, hidden=256, layers=2, bidirectional)
           → last timestep → (B, 512)
        6. Classifier: Dropout(0.5) → Linear(512, 2) → Softmax
    """

    VGG_FEAT_DIM = 512 * 7 * 7   # VGG19 spatial features before global pool
    POSE_DIM = 17 * 3             # 17 keypoints × (x, y, confidence)

    def __init__(self) -> None:
        super().__init__()

        # VGG19 backbone — features only (no classifier)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg_features = vgg.features    # (B, 512, 7, 7) output

        # Projection layers
        self.visual_proj = nn.Sequential(
            nn.Linear(self.VGG_FEAT_DIM, 256),
            nn.ReLU(inplace=True),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(self.POSE_DIM, 64),
            nn.ReLU(inplace=True),
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=320,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frame_features: (B, T, 3, 224, 224) — raw frame tensors
            poses:          (B, T, 51) — flattened pose keypoints

        Returns:
            (B, 2) softmax probabilities — class 0=normal, class 1=assault
        """
        B, T = frame_features.shape[:2]
        per_frame_feats = []

        for t in range(T):
            # Extract VGG features for this timestep
            x = frame_features[:, t]               # (B, 3, 224, 224)
            vgg_out = self.vgg_features(x)          # (B, 512, 7, 7)
            vgg_flat = vgg_out.view(B, -1)          # (B, VGG_FEAT_DIM)
            vis_emb = self.visual_proj(vgg_flat)    # (B, 256)

            # Pose embedding
            p = poses[:, t]                          # (B, 51)
            pose_emb = self.pose_proj(p)             # (B, 64)

            # Concatenate
            combined = torch.cat([vis_emb, pose_emb], dim=1)  # (B, 320)
            per_frame_feats.append(combined)

        # Stack into sequence tensor
        seq = torch.stack(per_frame_feats, dim=1)    # (B, T, 320)

        # BiLSTM
        lstm_out, _ = self.bilstm(seq)               # (B, T, 512)
        last_out = lstm_out[:, -1, :]                 # (B, 512)

        # Classifier
        logits = self.classifier(last_out)            # (B, 2)
        probs = F.softmax(logits, dim=-1)

        return probs


# ── Public detector class ─────────────────────────────────────────────────────

class AssaultDetector:
    """VGG19 + BiLSTM + Pose assault detector with per-track 16-frame buffers."""

    CLIP_LEN = 16
    FRAME_SIZE = (224, 224)
    ASSAULT_THRESHOLD = 0.70

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Build model
        print("[AssaultDetector] Loading VGG19 + BiLSTM assault model ...")
        self.net = _AssaultModel()
        self.net = self.net.to(self.device).half().eval()
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            self.net = torch.compile(self.net, mode="reduce-overhead")
            print(f"[{self.__class__.__name__}] torch.compile applied")
        print("[AssaultDetector] ✓ Loaded")

        # VGG feature extractor reference (used in predict for direct extraction)
        self.vgg_features = self.net.vgg_features

        # ImageNet normalisation
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Per-track rolling buffers
        self._frame_buffers: Dict[int, deque] = {}
        self._pose_buffers: Dict[int, deque] = {}

    # ── Push frame ────────────────────────────────────────────────────────────

    def push_frame(
        self,
        track_id: int,
        crop: np.ndarray,
        keypoints: np.ndarray,
    ) -> None:
        """
        Push a frame crop and its pose keypoints to the rolling buffers.

        Args:
            track_id:  Persistent track ID.
            crop:      BGR uint8 numpy array (person crop).
            keypoints: (17, 3) array — [x, y, conf] per COCO keypoint.
        """
        if track_id not in self._frame_buffers:
            self._frame_buffers[track_id] = deque(maxlen=self.CLIP_LEN)
            self._pose_buffers[track_id] = deque(maxlen=self.CLIP_LEN)

        # Resize crop to (224, 224), convert BGR → RGB → normalise → tensor
        resized = cv2.resize(crop, self.FRAME_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)   # (3, 224, 224) float32
        self._frame_buffers[track_id].append(tensor)

        # Flatten pose keypoints: (17, 3) → (51,)
        pose_flat = keypoints.flatten().astype(np.float32)[:51]
        # Pad if fewer than 51 values
        if len(pose_flat) < 51:
            pose_flat = np.pad(pose_flat, (0, 51 - len(pose_flat)))
        self._pose_buffers[track_id].append(pose_flat)

    # ── Predict ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, track_id: int) -> Optional[dict]:
        """
        Run assault detection on the 16-frame buffer for this track.

        Returns:
            None if either buffer has < 16 frames.
            Otherwise dict with keys:
              track_id, assault_confidence, is_assault
        """
        if track_id not in self._frame_buffers:
            return None
        if len(self._frame_buffers[track_id]) < self.CLIP_LEN:
            return None
        if len(self._pose_buffers[track_id]) < self.CLIP_LEN:
            return None

        # Stack 16 frame tensors → (1, 16, 3, 224, 224)
        frames = torch.stack(list(self._frame_buffers[track_id]))  # (16, 3, 224, 224)
        frames = frames.unsqueeze(0).to(self.device).half()         # (1, 16, 3, 224, 224)

        # Stack 16 pose vectors → (1, 16, 51)
        poses = np.stack(list(self._pose_buffers[track_id]))       # (16, 51)
        poses_t = torch.from_numpy(poses).unsqueeze(0).to(self.device).half()  # (1, 16, 51)

        # Forward pass (includes VGG extraction + BiLSTM + classifier)
        probs = self.net(frames, poses_t)                       # (1, 2)

        assault_confidence = float(probs[0][1].item())
        is_assault = assault_confidence >= self.ASSAULT_THRESHOLD

        return {
            "track_id": track_id,
            "assault_confidence": assault_confidence,
            "is_assault": is_assault,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_track(self, track_id: int) -> None:
        """Remove track_id from all buffers if present."""
        self._frame_buffers.pop(track_id, None)
        self._pose_buffers.pop(track_id, None)
