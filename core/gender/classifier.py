"""
Women Safety AI — CLIP + EfficientNet-B0 Gender Classifier
============================================================
Ensemble gender classifier combining:
  • CLIP ViT-B/32 (zero-shot: "a woman" vs "a man")
  • EfficientNet-B0 (pre-trained / fine-tunable, 2-class)

Both models run in FP16 on CUDA.  Results are cached per track_id
and recomputed only every 10 frames to avoid redundant inference.

Usage:
    clf = GenderClassifier()
    gender, conf = clf.classify(frame, bbox=[x1,y1,x2,y2], track_id=7)
"""

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
    """CLIP + EfficientNet-B0 ensemble for male / female / unknown."""

    LABELS = ["female", "male"]
    CONF_THRESHOLD = 0.75      # below this → "unknown"
    RECLASSIFY_EVERY = 10      # re-run inference every N frames per track

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
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            self.effnet = torch.compile(self.effnet, mode="reduce-overhead")
            print(f"[{self.__class__.__name__}] torch.compile applied (EfficientNet-B0)")

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
            # Pre-compute text features (done once, reused every call)
            text_tokens = clip.tokenize(["a woman", "a man"]).to(self.device)
            with torch.no_grad():
                self._text_feats = self.clip_model.encode_text(text_tokens)
                self._text_feats = self._text_feats / self._text_feats.norm(
                    dim=-1, keepdim=True
                )
        else:
            self.clip_model = None
            self.clip_preprocess = None
            self._text_feats = None

        # ── Caches ──────────────────────────────────────────────────────
        self._cache: Dict[int, Tuple[str, float]] = {}
        self._counters: Dict[int, int] = {}

    # ── public API ──────────────────────────────────────────────────────
    def classify(
        self,
        frame: np.ndarray,
        bbox: list,
        track_id: int,
    ) -> Tuple[str, float]:
        """
        Classify a tracked person's gender.

        Args:
            frame:    Full BGR frame from camera.
            bbox:     [x1, y1, x2, y2] bounding box.
            track_id: Persistent track ID from the tracker.

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

        # ── Ensemble: 50 / 50 weighted average ────────────────────────
        if clip_probs is not None:
            p_female = 0.5 * eff_probs[0] + 0.5 * clip_probs[0]
            p_male = 0.5 * eff_probs[1] + 0.5 * clip_probs[1]
        else:
            # CLIP not installed — EfficientNet only
            p_female = eff_probs[0]
            p_male = eff_probs[1]

        # ── Decision ───────────────────────────────────────────────────
        max_prob = max(p_female, p_male)
        if max_prob < self.CONF_THRESHOLD:
            result = ("unknown", float(max_prob))
        elif p_female >= p_male:
            result = ("female", float(p_female))
        else:
            result = ("male", float(p_male))

        self._cache[track_id] = result
        return result

    # ── private helpers ─────────────────────────────────────────────────
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
        return [float(probs[0]), float(probs[1])]
