#!/usr/bin/env python3
"""
Women Safety AI — Sprint 1 Live Demo
======================================
Runs the full Sprint 1 pipeline on a live USB webcam:
  • CameraStream → PersonTracker (YOLO + BoT-SORT) → GenderClassifier (CLIP + EfficientNet)

Displays an annotated window with:
  - Colour-coded bounding boxes (green=female, blue=male, grey=unknown)
  - Track ID and gender label above each box
  - FPS and person count overlay

Controls:
  - Press Q to quit

Usage:
    cd women_safety
    python scripts/run_s1_demo.py
"""

import sys
import os

# Ensure project root is on sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from core.camera.stream import CameraStream
from core.tracking.tracker import PersonTracker
from core.gender.classifier import GenderClassifier


# ── Colour map by gender ────────────────────────────────────────────────────
GENDER_COLOURS = {
    "female":  (0, 255, 0),     # green
    "male":    (255, 100, 0),   # blue-ish
    "unknown": (128, 128, 128), # grey
}


def draw_person(
    frame: np.ndarray,
    bbox: list,
    track_id: int,
    gender: str,
    gender_conf: float,
) -> None:
    """Draw a colour-coded bounding box with label on the frame (in-place)."""
    colour = GENDER_COLOURS.get(gender, (128, 128, 128))
    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label
    label = f"#{track_id} {gender} {gender_conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


def main() -> None:
    print("=" * 60)
    print("  Women Safety AI — Sprint 1 Demo")
    print("  Press Q to quit")
    print("=" * 60)

    # ── Initialise components ────────────────────────────────────────
    print("[1/3] Starting camera stream …")
    cam = CameraStream(source=0, camera_id=0).start()

    print("[2/3] Loading PersonTracker (YOLO11 + BoT-SORT) …")
    tracker = PersonTracker()

    print("[3/3] Loading GenderClassifier (CLIP + EfficientNet-B0) …")
    gender_clf = GenderClassifier()

    print("Ready! Showing live feed …\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # ── Detection + Tracking ─────────────────────────────────
            persons = tracker.update(frame)

            # ── Gender Classification ────────────────────────────────
            for person in persons:
                gender, conf = gender_clf.classify(
                    frame, person.bbox, person.track_id
                )
                person.gender = gender
                person.gender_conf = conf

            # ── Draw annotations ─────────────────────────────────────
            for person in persons:
                draw_person(
                    frame,
                    person.bbox,
                    person.track_id,
                    person.gender,
                    person.gender_conf,
                )

            # ── Overlay: FPS & person count ──────────────────────────
            overlay = f"FPS: {cam.fps} | Persons: {len(persons)}"
            cv2.putText(
                frame, overlay, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow("Sprint 1 — Demo", frame)

            # Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
