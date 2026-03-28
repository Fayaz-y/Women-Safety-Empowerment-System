"""
Women Safety AI — CLIP Zero-Shot Scene Context Scorer
======================================================
Uses CLIP ViT-B/32 to score full frames against threat and normal
prompts using zero-shot classification. Pre-computes text embeddings
at init and never recomputes them.

Threat prompts (index 0–2):
  "a woman being physically attacked"
  "a woman being harassed by a man"
  "a violent fight between people"

Normal prompts (index 3–4):
  "people walking normally on a street"
  "a normal public space with pedestrians"

The threat_score is the sum of softmax probabilities assigned to the
3 threat prompts.

Usage:
    ctx = CLIPContext()
    result = ctx.score(frame)
    print(result["clip_threat_score"], result["top_prompt"])
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip
except ImportError:
    clip = None

# ── Prompt constants ──────────────────────────────────────────────────────────

THREAT_PROMPTS = [
    "a woman being physically attacked",
    "a woman being harassed by a man",
    "a violent fight between people",
]

NORMAL_PROMPTS = [
    "people walking normally on a street",
    "a normal public space with pedestrians",
]

ALL_PROMPTS = THREAT_PROMPTS + NORMAL_PROMPTS  # length = 5


class CLIPContext:
    """CLIP zero-shot scene context scorer for threat assessment."""

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        if clip is None:
            raise ImportError(
                "CLIP is not installed. Please install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        # Load CLIP model
        print("[CLIPContext] Loading CLIP ViT-B/32 ...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print(f"[{self.__class__.__name__}] torch.compile applied")
        print("[CLIPContext] ✓ Loaded")

        # Pre-compute and normalise text features (fixed at init)
        tokens = clip.tokenize(ALL_PROMPTS).to(self.device)
        with torch.no_grad():
            self._text_feats = self.model.encode_text(tokens)
            self._text_feats = self._text_feats / self._text_feats.norm(
                dim=-1, keepdim=True
            )
        # Shape: (5, 512) — never recomputed

    # ── Main API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, frame: np.ndarray) -> dict:
        """
        Score the entire frame against threat and normal prompts.

        Args:
            frame: BGR uint8 numpy array (full camera frame).

        Returns:
            dict with keys:
              clip_threat_score (float 0–1): sum of probabilities for
                  the 3 threat prompts.
              top_prompt (str): highest-similarity prompt.
        """
        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Preprocess and encode image
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        image_feats = self.model.encode_image(image_tensor)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        # Cosine similarity → softmax over all 5 prompts
        similarity = (image_feats @ self._text_feats.T).squeeze(0)
        sims = F.softmax(similarity * 100.0, dim=0).float().cpu().numpy()

        # Threat score = sum of probabilities for the 3 threat prompts
        threat_score = float(sims[:3].sum())

        # Top prompt
        top_idx = int(np.argmax(sims))
        top_prompt = ALL_PROMPTS[top_idx]

        return {
            "clip_threat_score": threat_score,
            "top_prompt": top_prompt,
        }
