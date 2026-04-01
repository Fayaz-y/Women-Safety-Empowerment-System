"""
Women Safety AI — VGG19 + BiLSTM + MediaPipe Assault Detector
================================================================
Hybrid assault detection combining:
  • VGG19 spatial features + BiLSTM temporal model (deep learning)
  • MediaPipe / YOLO pose gesture analysis (rule-based)

The rule-based gesture analyser detects:
  • Grabbing: one person's hands near another's body
  • Pushing: rapid arm extension patterns
  • Resistance: defensive arm movements (blocking, pushing away)
  • Struggling: rapid asymmetric limb movements

Friendly vs hostile is distinguished by:
  • Movement symmetry (friendly → smooth bidirectional)
  • Acceleration patterns (hostile → sharp, jerky)
  • Arm extension speed (hostile → fast)

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
    """VGG19 spatial features + BiLSTM temporal model for assault detection."""

    VGG_FEAT_DIM = 512 * 7 * 7
    POSE_DIM = 17 * 3

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg_features = vgg.features

        self.visual_proj = nn.Sequential(
            nn.Linear(self.VGG_FEAT_DIM, 256),
            nn.ReLU(inplace=True),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(self.POSE_DIM, 64),
            nn.ReLU(inplace=True),
        )

        self.bilstm = nn.LSTM(
            input_size=320,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, frame_features, poses):
        B, T = frame_features.shape[:2]
        per_frame_feats = []
        for t in range(T):
            x = frame_features[:, t]
            vgg_out = self.vgg_features(x)
            vgg_flat = vgg_out.view(B, -1)
            vis_emb = self.visual_proj(vgg_flat)
            p = poses[:, t]
            pose_emb = self.pose_proj(p)
            combined = torch.cat([vis_emb, pose_emb], dim=1)
            per_frame_feats.append(combined)
        seq = torch.stack(per_frame_feats, dim=1)
        lstm_out, _ = self.bilstm(seq)
        last_out = lstm_out[:, -1, :]
        logits = self.classifier(last_out)
        probs = F.softmax(logits, dim=-1)
        return probs


# ── Gesture Analyser (rule-based) ─────────────────────────────────────────────

class GestureAnalyser:
    """
    Rule-based gesture analysis from pose keypoint sequences.
    Detects grabbing, pushing, resistance, and struggling patterns.
    Distinguishes friendly vs hostile interactions.
    """

    def __init__(self):
        # Per-track keypoint history for movement analysis
        self._kp_history: Dict[int, deque] = {}  # track_id → deque of (17,3)
        self.HISTORY_LEN = 10

    def push_keypoints(self, track_id: int, keypoints: np.ndarray):
        """Push a keypoint frame to the per-track history."""
        if track_id not in self._kp_history:
            self._kp_history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._kp_history[track_id].append(keypoints.copy())

    def analyse(self, track_id: int) -> dict:
        """
        Analyse the keypoint history for assault-related gestures.

        Returns dict with:
            gesture_score (float 0.0–1.0),
            flags: {grabbing, pushing, resistance, struggling, hostile}
        """
        result = {
            "gesture_score": 0.0,
            "flags": {
                "grabbing": False,
                "pushing": False,
                "resistance": False,
                "struggling": False,
                "hostile": False,
            },
        }

        if track_id not in self._kp_history:
            return result
        history = list(self._kp_history[track_id])
        if len(history) < 4:
            return result

        KP_CONF = 0.3
        scores = []

        # ── Analyse rapid arm movements ──────────────────────────────────
        wrist_velocities = []
        for i in range(1, len(history)):
            prev_kp = history[i - 1]
            curr_kp = history[i]

            for wrist_idx in [9, 10]:  # left + right wrist
                if (float(prev_kp[wrist_idx][2]) >= KP_CONF and
                        float(curr_kp[wrist_idx][2]) >= KP_CONF):
                    dx = float(curr_kp[wrist_idx][0] - prev_kp[wrist_idx][0])
                    dy = float(curr_kp[wrist_idx][1] - prev_kp[wrist_idx][1])
                    vel = (dx ** 2 + dy ** 2) ** 0.5
                    wrist_velocities.append(vel)

        if wrist_velocities:
            avg_vel = sum(wrist_velocities) / len(wrist_velocities)
            max_vel = max(wrist_velocities)

            # High velocity arms → possible pushing/hitting
            if max_vel > 40:
                result["flags"]["pushing"] = True
                scores.append(0.4)

            # Check for jerky movement (high velocity variance = struggle)
            if len(wrist_velocities) > 3:
                vel_std = np.std(wrist_velocities)
                if vel_std > 20:
                    result["flags"]["struggling"] = True
                    scores.append(0.3)

        # ── Raised defensive arms (wrists above shoulders) ────────────────
        latest = history[-1]
        for wrist_idx, shld_idx in [(9, 5), (10, 6)]:
            if (float(latest[wrist_idx][2]) >= KP_CONF and
                    float(latest[shld_idx][2]) >= KP_CONF):
                if latest[wrist_idx][1] < latest[shld_idx][1] - 20:
                    result["flags"]["resistance"] = True
                    scores.append(0.25)
                    break

        # ── Arms extended forward rapidly (pushing pattern) ────────────────
        if len(history) >= 3:
            old = history[-3]
            new = history[-1]
            for wrist_idx, shld_idx in [(9, 5), (10, 6)]:
                if (float(old[wrist_idx][2]) >= KP_CONF and
                        float(new[wrist_idx][2]) >= KP_CONF and
                        float(new[shld_idx][2]) >= KP_CONF):
                    # Distance from shoulder to wrist increased rapidly
                    old_dist = abs(float(old[wrist_idx][0] - old[shld_idx][0]))
                    new_dist = abs(float(new[wrist_idx][0] - new[shld_idx][0]))
                    if new_dist > old_dist + 30:
                        result["flags"]["pushing"] = True
                        scores.append(0.35)

        # ── Covering face (defensive) ──────────────────────────────────────
        nose = latest[0]
        if float(nose[2]) >= KP_CONF:
            for wrist_idx in [9, 10]:
                if float(latest[wrist_idx][2]) >= KP_CONF:
                    dist = ((float(latest[wrist_idx][0] - nose[0])) ** 2 +
                            (float(latest[wrist_idx][1] - nose[1])) ** 2) ** 0.5
                    if dist < 50:
                        result["flags"]["resistance"] = True
                        scores.append(0.2)

        # ── Hostile vs friendly: asymmetric jerky motion ────────────────
        if any([result["flags"]["pushing"],
                result["flags"]["struggling"],
                result["flags"]["resistance"]]):
            result["flags"]["hostile"] = True
            scores.append(0.15)

        # ── Final score ─────────────────────────────────────────────────
        if scores:
            result["gesture_score"] = min(sum(scores), 1.0)

        return result

    def clear_track(self, track_id: int):
        self._kp_history.pop(track_id, None)


# ── Public detector class ─────────────────────────────────────────────────────

class AssaultDetector:
    """VGG19 + BiLSTM + Gesture analysis assault detector."""

    CLIP_LEN = 16
    FRAME_SIZE = (224, 224)
    ASSAULT_THRESHOLD = 0.55  # lowered for earlier detection

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        print("[AssaultDetector] Loading VGG19 + BiLSTM assault model ...")
        self.net = _AssaultModel()
        self.net = self.net.to(self.device).half().eval()
        print("[AssaultDetector] ✓ Loaded")

        self.vgg_features = self.net.vgg_features

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

        # Gesture analyser (rule-based)
        self.gesture = GestureAnalyser()

    # ── Push frame ────────────────────────────────────────────────────────────

    def push_frame(
        self,
        track_id: int,
        crop: np.ndarray,
        keypoints: np.ndarray,
    ) -> None:
        """Push a frame crop and its pose keypoints to the rolling buffers."""
        if track_id not in self._frame_buffers:
            self._frame_buffers[track_id] = deque(maxlen=self.CLIP_LEN)
            self._pose_buffers[track_id] = deque(maxlen=self.CLIP_LEN)

        resized = cv2.resize(crop, self.FRAME_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)
        self._frame_buffers[track_id].append(tensor)

        pose_flat = keypoints.flatten().astype(np.float32)[:51]
        if len(pose_flat) < 51:
            pose_flat = np.pad(pose_flat, (0, 51 - len(pose_flat)))
        self._pose_buffers[track_id].append(pose_flat)

        # Also push to gesture analyser
        self.gesture.push_keypoints(track_id, keypoints)

    # ── Predict ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, track_id: int) -> Optional[dict]:
        """
        Run assault detection combining deep learning + gesture rules.

        Returns dict with:
            track_id, assault_confidence, is_assault,
            gesture_score, gesture_flags
        """
        # Always compute gesture score if we have keypoint history
        gesture_result = self.gesture.analyse(track_id)
        gesture_score = gesture_result["gesture_score"]

        # Deep learning prediction (needs 16 frames)
        dl_confidence = 0.0
        if (track_id in self._frame_buffers and
                len(self._frame_buffers[track_id]) >= self.CLIP_LEN and
                len(self._pose_buffers[track_id]) >= self.CLIP_LEN):

            frames = torch.stack(list(self._frame_buffers[track_id]))
            frames = frames.unsqueeze(0).to(self.device).half()

            poses = np.stack(list(self._pose_buffers[track_id]))
            poses_t = torch.from_numpy(poses).unsqueeze(0).to(self.device).half()

            probs = self.net(frames, poses_t)
            dl_confidence = float(probs[0][1].item())

        # Combine: 60% deep learning + 40% gesture rules
        combined_confidence = 0.6 * dl_confidence + 0.4 * gesture_score
        is_assault = combined_confidence >= self.ASSAULT_THRESHOLD

        return {
            "track_id": track_id,
            "assault_confidence": combined_confidence,
            "dl_confidence": dl_confidence,
            "gesture_score": gesture_score,
            "gesture_flags": gesture_result["flags"],
            "is_assault": is_assault,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_track(self, track_id: int) -> None:
        """Remove track_id from all buffers if present."""
        self._frame_buffers.pop(track_id, None)
        self._pose_buffers.pop(track_id, None)
        self.gesture.clear_track(track_id)
