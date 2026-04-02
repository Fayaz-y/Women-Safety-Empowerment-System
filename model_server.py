"""
Women Safety AI — Remote Model Inference Server
================================================
Loads all models exactly like the backend does.
Provides REST endpoints for remote inference via ngrok.

Run with:
    python model_server.py
    
Then expose via ngrok:
    ngrok http 9000
"""

import base64
import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from core.gender.classifier import GenderClassifier
from core.pose.estimator import PoseEstimator
from core.proximity.engine import ProximityEngine
from core.violence.detector import ViolenceDetector
from core.assault.detector import AssaultDetector
from core.anomaly.detector import AnomalyDetector
from core.context.clip_context import CLIPContext
from core.fusion.fusion import FusionLayer


# ── Models Dictionary ──
MODELS = {
    "gender": None,
    "pose": None,
    "proximity": None,
    "violence": None,
    "assault": None,
    "anomaly": None,
    "clip": None,
    "fusion": None,
}


# ── Request/Response Models ──
class InferenceRequest(BaseModel):
    """Generic inference request."""
    image_base64: str  # Base64 encoded image


class InferenceResponse(BaseModel):
    """Generic inference response."""
    status: str
    data: dict = {}
    error: str = None


# ── Load Models on Startup (Same as PipelineEngine) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""
    print("\n" + "="*70)
    print("Women Safety AI — Model Inference Server")
    print("="*70 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    def _vram_gb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    try:
        # 1. Gender classifier
        print("[1/8] GenderClassifier ...")
        MODELS["gender"] = GenderClassifier(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 2. Pose estimator
        print("[2/8] PoseEstimator ...")
        MODELS["pose"] = PoseEstimator(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 3. Proximity engine
        print("[3/8] ProximityEngine ...")
        MODELS["proximity"] = ProximityEngine()
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 4. Violence detector
        print("[4/8] ViolenceDetector ...")
        MODELS["violence"] = ViolenceDetector(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 5. Assault detector
        print("[5/8] AssaultDetector ...")
        MODELS["assault"] = AssaultDetector(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 6. Anomaly detector
        print("[6/8] AnomalyDetector ...")
        MODELS["anomaly"] = AnomalyDetector(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 7. CLIP context (includes gender classifier)
        print("[7/8] CLIPContext ...")
        MODELS["clip"] = CLIPContext(device=device)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # 8. Fusion layer
        print("[8/8] FusionLayer ...")
        MODELS["fusion"] = FusionLayer(threshold=settings.fusion_threshold)
        print(f"      ✓ VRAM: {_vram_gb():.2f} GB")

        # Summary
        vram = _vram_gb()
        print(f"\n✓ All models loaded! Total VRAM: {vram:.2f} GB\n")
        
    except Exception as e:
        print(f"\n✗ Error loading models: {e}\n")
        raise

    yield

    # Cleanup on shutdown
    print("[Model Server] Shutting down...")


app = FastAPI(
    title="Women Safety AI — Model Inference Server",
    version="1.0.0",
    description="Remote inference endpoint for model computations",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check."""
    models_loaded = sum(1 for m in MODELS.values() if m is not None)
    return {
        "service": "Women Safety AI — Model Inference Server",
        "version": "1.0.0",
        "models_loaded": f"{models_loaded}/{len(MODELS)}",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/api/v1/health")
def health_check():
    """Check which models are loaded."""
    return {
        "status": "ok",
        "models": {k: v is not None for k, v in MODELS.items()},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting on http://0.0.0.0:9000")
    print("Expose with: ngrok http 9000\n")
    
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
        workers=1,
    )
