"""
Women Safety AI — CLIP + Body-Proportion Gender Classifier
=============================================================
Ensemble gender classifier combining:
  • CLIP ViT-B/32 (zero-shot: "a woman" vs "a man")  — weight 0.55
  • Body-proportion heuristics from pose keypoints     — weight 0.30
  • EfficientNet-B0 (pretrained feature extractor)     — weight 0.15

Designed for CCTV surveillance angles where faces may not be visible.
Body proportions (shoulder width, hip width, height ratios) provide
robust gender signals from any camera angle.

Results use temporal voting over 30 frames for stable classification.

Usage:
    clf = GenderClassifier()
    gender, conf = clf.classify(frame, bbox=[x1,y1,x2,y2], track_id=7,
                                 keypoints=kp_array)
"""

from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import clip
except ImportError:
    clip = None  # graceful fallback — CLIP not installed

import timm


class GenderClassifier:
    """CLIP + Body-proportion + EfficientNet ensemble for gender classification."""

    LABELS = ["female", "male"]
    CONF_THRESHOLD = 0.55           # lowered — body proportions compensate
    RECLASSIFY_EVERY = 5            # more frequent than before for accuracy
    TEMPORAL_WINDOW = 30            # frames for temporal voting

    # Ensemble weights
    W_CLIP = 0.55
    W_BODY = 0.30
    W_EFFNET = 0.15

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # ── EfficientNet-B0 ─────────────────────────────────────────────
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=2
        )
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.effnet.load_state_dict(state)
        self.effnet = self.effnet.to(self.device)
        if self.device == "cuda":
            self.effnet = self.effnet.half()
        self.effnet.eval()

        # ImageNet normalisation transform for EfficientNet
        self._eff_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ── CLIP ViT-B/32 ──────────────────────────────────────────────
        self._clip_available = clip is not None
        if self._clip_available:
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )
            self.clip_model.eval()
            # Enhanced prompts for full-body CCTV views
            prompts = [
                "a woman walking",
                "a female person standing",
                "a man walking",
                "a male person standing",
            ]
            text_tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                self._text_feats = self.clip_model.encode_text(text_tokens)
                self._text_feats = self._text_feats / self._text_feats.norm(
                    dim=-1, keepdim=True
                )
        else:
            self.clip_model = None
            self.clip_preprocess = None
            self._text_feats = None

        # ── Caches & temporal voting ─────────────────────────────────────
        self._cache: Dict[int, Tuple[str, float]] = {}
        self._counters: Dict[int, int] = {}
        self._vote_history: Dict[int, deque] = {}  # track_id → deque of (p_female, p_male)

    # ── public API ──────────────────────────────────────────────────────
    def classify(
        self,
        frame: np.ndarray,
        bbox: list,
        track_id: int,
        keypoints: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """
        Classify a tracked person's gender.

        Args:
            frame:     Full BGR frame from camera.
            bbox:      [x1, y1, x2, y2] bounding box.
            track_id:  Persistent track ID from the tracker.
            keypoints: Optional (17, 3) pose keypoints for body proportions.

        Returns:
            (gender, confidence) — gender is "female", "male", or "unknown".
        """
        # Increment counter
        self._counters[track_id] = self._counters.get(track_id, 0) + 1
        counter = self._counters[track_id]

        # Return cached result unless it's time to re-classify
        if counter % self.RECLASSIFY_EVERY != 0 and track_id in self._cache:
            return self._cache[track_id]

        # ── Crop person from frame (with 5px padding) ──────────────────
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]) - 5)
        y1 = max(0, int(bbox[1]) - 5)
        x2 = min(w, int(bbox[2]) + 5)
        y2 = min(h, int(bbox[3]) + 5)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            result = ("unknown", 0.0)
            self._cache[track_id] = result
            return result

        # BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # ── EfficientNet forward pass ──────────────────────────────────
        eff_probs = self._run_efficientnet(pil_img)

        # ── CLIP forward pass ──────────────────────────────────────────
        clip_probs = self._run_clip(pil_img)

        # ── Body-proportion heuristics ─────────────────────────────────
        body_probs = self._body_proportions(keypoints, bbox)

        # ── Weighted ensemble ──────────────────────────────────────────
        p_female = self.W_EFFNET * eff_probs[0]
        p_male = self.W_EFFNET * eff_probs[1]

        if clip_probs is not None:
            p_female += self.W_CLIP * clip_probs[0]
            p_male += self.W_CLIP * clip_probs[1]
        else:
            # Without CLIP, redistribute weight
            p_female += self.W_CLIP * eff_probs[0]
            p_male += self.W_CLIP * eff_probs[1]

        if body_probs is not None:
            p_female += self.W_BODY * body_probs[0]
            p_male += self.W_BODY * body_probs[1]
        else:
            # Without keypoints, redistribute
            if clip_probs is not None:
                p_female += self.W_BODY * clip_probs[0]
                p_male += self.W_BODY * clip_probs[1]
            else:
                p_female += self.W_BODY * eff_probs[0]
                p_male += self.W_BODY * eff_probs[1]

        # ── Temporal voting ────────────────────────────────────────────
        if track_id not in self._vote_history:
            self._vote_history[track_id] = deque(maxlen=self.TEMPORAL_WINDOW)
        self._vote_history[track_id].append((p_female, p_male))

        # Average over temporal window
        votes = self._vote_history[track_id]
        avg_female = sum(v[0] for v in votes) / len(votes)
        avg_male = sum(v[1] for v in votes) / len(votes)

        # ── Decision ───────────────────────────────────────────────────
        max_prob = max(avg_female, avg_male)
        if max_prob < self.CONF_THRESHOLD:
            result = ("unknown", float(max_prob))
        elif avg_female >= avg_male:
            result = ("female", float(avg_female))
        else:
            result = ("male", float(avg_male))

        self._cache[track_id] = result
        return result

    # ── private helpers ─────────────────────────────────────────────────

    def _body_proportions(
        self, keypoints: Optional[np.ndarray], bbox: list
    ) -> Optional[list]:
        """
        Estimate gender from body proportions using pose keypoints.

        Uses shoulder-to-hip ratio, torso proportions, and overall
        body shape — works regardless of camera angle.

        Returns [P(female), P(male)] or None if keypoints unavailable.
        """
        if keypoints is None or len(keypoints) < 17:
            return None

        KP_CONF = 0.4

        def kp_valid(idx):
            return float(keypoints[idx][2]) >= KP_CONF

        def kp_xy(idx):
            return float(keypoints[idx][0]), float(keypoints[idx][1])

        # Need shoulders and hips
        if not (kp_valid(5) and kp_valid(6) and kp_valid(11) and kp_valid(12)):
            return None

        l_shoulder = kp_xy(5)
        r_shoulder = kp_xy(6)
        l_hip = kp_xy(11)
        r_hip = kp_xy(12)

        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        hip_width = abs(l_hip[0] - r_hip[0])

        if shoulder_width < 5 or hip_width < 5:
            return None

        # Shoulder-to-hip ratio: males ~1.3+, females ~1.0 or lower
        sh_ratio = shoulder_width / hip_width

        # Torso length (avg shoulder y to avg hip y)
        torso_len = abs(
            (l_shoulder[1] + r_shoulder[1]) / 2 -
            (l_hip[1] + r_hip[1]) / 2
        )

        # Bbox aspect ratio (height / width)
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        aspect = bbox_h / max(bbox_w, 1)

        # Score computation
        # Males: wider shoulders relative to hips, boxier build
        # Females: more equal or wider hips, taller aspect ratio
        male_score = 0.0

        if sh_ratio > 1.25:
            male_score += 0.4
        elif sh_ratio > 1.10:
            male_score += 0.2
        elif sh_ratio < 0.95:
            male_score -= 0.3  # wider hips → female indicator

        # Wider build (lower aspect ratio) → male indicator
        if aspect < 2.5:
            male_score += 0.15
        elif aspect > 3.2:
            male_score -= 0.15

        # Shoulder width relative to bbox width
        shoulder_ratio = shoulder_width / max(bbox_w, 1)
        if shoulder_ratio > 0.4:
            male_score += 0.15
        elif shoulder_ratio < 0.28:
            male_score -= 0.15

        # Convert to probability
        p_male = 0.5 + male_score
        p_male = max(0.1, min(0.9, p_male))
        p_female = 1.0 - p_male

        return [p_female, p_male]

    @torch.no_grad()
    def _run_efficientnet(self, pil_img: Image.Image) -> list:
        """Return [P(female), P(male)] from EfficientNet-B0."""
        tensor = self._eff_transform(pil_img).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
        logits = self.effnet(tensor)
        probs = F.softmax(logits, dim=1)[0].float().cpu().numpy()
        return [float(probs[0]), float(probs[1])]

    @torch.no_grad()
    def _run_clip(self, pil_img: Image.Image) -> Optional[list]:
        """Return [P(woman), P(man)] from CLIP, or None if unavailable."""
        if not self._clip_available:
            return None
        image_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        image_feats = self.clip_model.encode_image(image_tensor)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        similarity = (image_feats @ self._text_feats.T).squeeze(0)
        probs = F.softmax(similarity * 100.0, dim=0).float().cpu().numpy()
        # prompts 0,1 = female; prompts 2,3 = male → average each pair
        p_female = float((probs[0] + probs[1]) / 2)
        p_male = float((probs[2] + probs[3]) / 2)
        # Renormalize
        total = p_female + p_male
        if total > 0:
            p_female /= total
            p_male /= total
        return [p_female, p_male]
