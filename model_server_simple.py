"""
Women Safety AI — Remote Model Inference Server (SIMPLIFIED)
=============================================================
Lightweight FastAPI server that provides model inference endpoints.

Run with:
    python model_server.py
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

from core.detection.detector import PersonDetector


# ── Models Dictionary ──
MODELS = {
    "detection": None,
}


# ── Request/Response Models ──
class DetectionRequest(BaseModel):
    """Generic detection request."""
    image_base64: str  # Base64 encoded image
    confidence_threshold: float = 0.5


class InferenceResponse(BaseModel):
    """Generic inference response."""
    status: str
    data: dict = {}
    error: str = None


# ── Load Models on Startup ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Model Server] Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model Server] Using device: {device}")

    try:
        # Load detection model (PersonDetector)
        print("[Model Server] Loading YOLO person detection model...")
        MODELS["detection"] = PersonDetector(device=device, model_name="yolo11n.pt")
        print("[Model Server] ✓ Detection model loaded successfully!")
        print("[Model Server] ✓ All models ready!\n")
        
    except Exception as e:
        print(f"[Model Server] ✗ Error loading models: {e}")
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


@app.post("/api/v1/detect")
async def detect_persons(request: DetectionRequest):
    """
    Detect persons in image.

    Args:
        request.image_base64: Base64 encoded image
        request.confidence_threshold: Detection confidence threshold

    Returns:
        List of detections with boxes and confidence scores
    """
    if MODELS["detection"] is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        # Run detection (PersonDetector.detect returns List[Detection])
        detections = MODELS["detection"].detect(image)

        # Format response
        results = []
        for det in detections:
            results.append({
                "class_id": 0,  # Always person
                "class_name": "person",
                "confidence": float(det.confidence),
                "box": {
                    "x1": float(det.bbox[0]),
                    "y1": float(det.bbox[1]),
                    "x2": float(det.bbox[2]),
                    "y2": float(det.bbox[3]),
                },
            })

        return InferenceResponse(status="success", data={"detections": results})

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/health")
def health_check():
    """Check which models are loaded."""
    return {
        "status": "ok",
        "models": {
            "detection": MODELS["detection"] is not None,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Women Safety AI — Model Inference Server")
    print("="*70)
    print("Starting on http://0.0.0.0:9000")
    print("="*70 + "\n")
    
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
        workers=1,
    )
