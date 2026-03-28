"""
Women Safety AI — YOLO11 Person Detector
==========================================
Detects persons (class 0) using Ultralytics YOLO11. Runs in FP16 on CUDA.
Returns a list of Detection dataclass instances.

Usage:
    detector = PersonDetector()
    detections = detector.detect(frame)
    annotated = detector.draw(frame, detections)
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Single person detection result."""

    bbox: List[float]         # [x1, y1, x2, y2] in pixels (xyxy)
    confidence: float         # 0.0 – 1.0
    class_id: int = 0         # always 0 (person)


class PersonDetector:
    """YOLO11 person-only detector with FP16 CUDA inference."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.50,
    ):
        # Fall back to CPU if CUDA unavailable
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.conf = conf_threshold

        # Load model
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # FP16 only on CUDA
        self._half = self.device == "cuda"

        # torch.compile for faster inference (if available)
        import sys
        if hasattr(torch, "compile") and sys.platform != "win32":
            try:
                self.model.model = torch.compile(
                    self.model.model, mode="reduce-overhead"
                )
                print(f"[{self.__class__.__name__}] torch.compile applied")
            except Exception:
                pass  # Ultralytics may not always support direct compile

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on a BGR frame.

        Returns:
            List of Detection instances (empty if no persons found).
        """
        results = self.model(
            frame,
            conf=self.conf,
            classes=[0],      # person class only
            verbose=False,
            half=self._half,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(Detection(bbox=xyxy, confidence=conf))

        return detections

    def draw(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw detection boxes on a frame copy.

        Returns:
            Annotated frame (never modifies the original).
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            # Green rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label
            label = f"person {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated
