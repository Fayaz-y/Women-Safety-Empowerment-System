"""
Women Safety AI — Remote Model Inference Server
================================================
This standalone FastAPI server runs on YOUR laptop with GPU and provides
model inference endpoints. The backend (running elsewhere) calls these
endpoints via Cloudflare tunneling.

Run with:
    python model_server.py
    
Or with uvicorn directly:
    python -m uvicorn model_server:app --host 0.0.0.0 --port 9000

Then expose via Cloudflare:
    cloudflared tunnel create women-safety-model
    # Configure to forward to http://localhost:9000
"""

import io
import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.detection.detector import PersonDetector
from core.assault.detector import AssaultDetector
from core.gender.classifier import GenderClassifier
from core.proximity.engine import ProximityEngine
from core.pose.estimator import PoseEstimator


# ── Models Dictionary ──
MODELS = {
    "detection": None,
    "assault": None,
    "gender": None,
    "pose": None,
    "proximity": None,
}


# ── Request/Response Models ──
class DetectionRequest(BaseModel):
    """Generic detection request."""
    image_base64: str = None  # Base64 encoded image
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
        # Load detection model (YOLOv11)
        print("[Model Server] Loading YOLO detection model...")
        MODELS["detection"] = PersonDetector(device=device, model_name="yolo11n.pt")

        # Load assault detection model
        print("[Model Server] Loading assault detection model...")
        MODELS["assault"] = AssaultDetector(device=device)

        # Load gender classifier
        print("[Model Server] Loading gender classifier...")
        MODELS["gender"] = GenderClassifier(device=device)

        # Load pose estimator
        print("[Model Server] Loading pose estimator...")
        MODELS["pose"] = PoseEstimator(device=device)

        # Initialize proximity engine (stateless but needs config)
        print("[Model Server] Initializing proximity engine...")
        MODELS["proximity"] = ProximityEngine(radius_px=150)

        print("[Model Server] ✓ All models loaded successfully!\n")
        
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
    allow_origins=["*"],  # Allow all origins (internal use only)
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
        "models_loaded": f"{models_loaded}/5",
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
        import base64
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
                "confidence": det.confidence,
                "box": {
                    "x1": det.bbox[0],
                    "y1": det.bbox[1],
                    "x2": det.bbox[2],
                    "y2": det.bbox[3],
                },
            })

        return InferenceResponse(status="success", data={"detections": results})

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/pose")
async def estimate_pose(request: DetectionRequest):
    """
    Estimate pose (keypoints) from image.

    Returns:
        List of pose keypoints with confidence scores
    """
    if MODELS["pose"] is None:
        raise HTTPException(status_code=503, detail="Pose estimation model not loaded")

    try:
        import base64
        image_data = base64.b64decode(request.image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        # Run pose estimation
        poses = MODELS["pose"].infer(image)

        # Format response
        results = []
        for pose in poses:
            results.append({
                "keypoints": pose["keypoints"].tolist() if hasattr(pose.get("keypoints"), "tolist") else pose.get("keypoints"),
                "confidence": float(pose.get("confidence", 0.0)),
            })

        return InferenceResponse(status="success", data={"poses": results})

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/gender")
async def classify_gender(request: DetectionRequest):
    """
    Classify gender from person image.

    Returns:
        Gender classification (male/female) with confidence
    """
    if MODELS["gender"] is None:
        raise HTTPException(status_code=503, detail="Gender classifier not loaded")

    try:
        import base64
        image_data = base64.b64decode(request.image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        # Run gender classification
        result = MODELS["gender"].infer(image)

        return InferenceResponse(status="success", data=result)

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/assault")
async def detect_assault(request: DetectionRequest):
    """
    Detect assault/violence in image.

    Returns:
        Assault detection score
    """
    if MODELS["assault"] is None:
        raise HTTPException(status_code=503, detail="Assault detector not loaded")

    try:
        import base64
        image_data = base64.b64decode(request.image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        # Run assault detection
        score = MODELS["assault"].infer(image)

        return InferenceResponse(status="success", data={"assault_score": float(score)})

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/proximity")
async def analyze_proximity(proximity_data: dict):
    """
    Analyze proximity between detected persons.

    Args:
        proximity_data: Dictionary with detections and other relevant data

    Returns:
        Proximity analysis results
    """
    if MODELS["proximity"] is None:
        raise HTTPException(status_code=503, detail="Proximity engine not loaded")

    try:
        # The proximity engine typically works with detection results
        results = MODELS["proximity"].analyze(proximity_data)
        return InferenceResponse(status="success", data=results)

    except Exception as e:
        return InferenceResponse(status="error", error=str(e))


@app.post("/api/v1/health")
def health_check():
    """Check which models are loaded."""
    return {
        "status": "ok",
        "models": {
            "detection": MODELS["detection"] is not None,
            "assault": MODELS["assault"] is not None,
            "gender": MODELS["gender"] is not None,
            "pose": MODELS["pose"] is not None,
            "proximity": MODELS["proximity"] is not None,
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
