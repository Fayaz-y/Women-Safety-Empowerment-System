#!/usr/bin/env python3
"""
Women Safety AI — Sprint 2 Live Demo
======================================
Runs the full Sprint 2 pipeline on a live USB webcam:

  CameraStream → PersonTracker (BoT-SORT)
      → GenderClassifier (CLIP + EfficientNet)
      → PoseEstimator (YOLO11-Pose + distress analysis)
      → ProximityEngine (lone-woman + proximity violation)

Display annotations:
  • Gender-coloured bounding boxes (Sprint 1 style)
  • LONE WOMAN  → yellow border + label
  • Skeleton overlay (red when distress detected, green otherwise)
  • Proximity violation → console alert + red cross overlay on woman
  • FPS + person count + proximity violation counter overlay

Controls:
  Q → quit

Usage:
    cd women_safety
    python scripts/run_s2_demo.py
"""

import sys
import os
import time

# ── Ensure project root is importable ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from core.camera.stream import CameraStream
from core.tracking.tracker import PersonTracker, TrackedPerson
from core.gender.classifier import GenderClassifier
from core.pose.estimator import PoseEstimator, PoseResult
from core.proximity.engine import ProximityEngine


# ── Colour palette ────────────────────────────────────────────────────────────
GENDER_COLOURS = {
    "female":  (0, 255, 0),      # green
    "male":    (255, 100, 0),    # blue-ish
    "unknown": (128, 128, 128),  # grey
}
LONE_WOMAN_COLOUR = (0, 220, 255)   # yellow-ish (BGR)
ALERT_COLOUR      = (0, 0, 255)     # red


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_person_box(
    frame: np.ndarray,
    person: TrackedPerson,
    pose: PoseResult | None = None,
) -> None:
    """Draw bbox, label, and (optionally) skeleton for one person."""
    bbox = person.bbox
    x1, y1, x2, y2 = [int(c) for c in bbox]

    if person.is_lone_woman:
        # Yellow double-border for lone woman
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), LONE_WOMAN_COLOUR, 3)
        label = f"#{person.track_id} LONE WOMAN {person.isolation_seconds:.1f}s"
        label_colour = LONE_WOMAN_COLOUR
    else:
        colour = GENDER_COLOURS.get(person.gender, (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"#{person.track_id} {person.gender} {person.gender_conf:.0%}"
        label_colour = colour

    # Add distress score to label if pose is available
    if pose is not None:
        label += f"  D:{pose.distress_score:.2f}"

    # Label background + text
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), label_colour, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
    )


def draw_violation_flash(
    frame: np.ndarray,
    woman_bbox: list,
) -> None:
    """Draw a red X over the woman's bbox to signal a proximity violation."""
    x1, y1, x2, y2 = [int(c) for c in woman_bbox]
    cv2.line(frame, (x1, y1), (x2, y2), ALERT_COLOUR, 3, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x1, y2), ALERT_COLOUR, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_COLOUR, 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Women Safety AI — Sprint 2 Demo")
    print("  Pose + Proximity + Lone-Woman Detection")
    print("  Press Q to quit")
    print("=" * 65)

    # ── 1. Camera ────────────────────────────────────────────────────────
    print("\n[1/4] Starting camera stream (source=0) …")
    cam = CameraStream(source=0, camera_id=0).start()

    # ── 2. Tracker ───────────────────────────────────────────────────────
    # NOTE: PersonTracker still uses yolo11n.pt (detection only).
    # PoseEstimator below uses yolo11n-pose.pt (same tracking persist=True,
    # so track IDs from both models will align when run together).
    print("[2/4] Loading PersonTracker (YOLO11n + BoT-SORT) …")
    tracker = PersonTracker()

    # ── 3. Gender classifier ─────────────────────────────────────────────
    print("[3/4] Loading GenderClassifier (CLIP + EfficientNet-B0) …")
    gender_clf = GenderClassifier()

    # ── 4. Pose estimator & proximity engine ─────────────────────────────
    print("[4/4] Loading PoseEstimator (YOLO11n-Pose) + ProximityEngine …")
    pose_est  = PoseEstimator()           # auto-downloads yolo11n-pose.pt
    proximity = ProximityEngine()

    print("\nReady! Showing live feed …\n")

    # Counters for overlay
    violation_total = 0

    try:
        while True:
            t0 = time.perf_counter()

            frame = cam.read()
            if frame is None:
                continue

            vis = frame.copy()   # annotated display copy

            # ── Detection + Tracking ─────────────────────────────────────
            persons = tracker.update(frame)

            # ── Pose estimation ──────────────────────────────────────────
            pose_results = pose_est.estimate(frame)
            pose_map: dict[int, PoseResult] = {
                pr.track_id: pr for pr in pose_results
            }

            # ── Gender classification ────────────────────────────────────
            for person in persons:
                gender, conf = gender_clf.classify(
                    frame, person.bbox, person.track_id
                )
                person.gender     = gender
                person.gender_conf = conf

            # ── Proximity engine ─────────────────────────────────────────
            events = proximity.update(persons)

            # Console alerts for violations
            for ev in events:
                violation_total += 1
                print(
                    f"[ALERT] Proximity violation: "
                    f"woman #{ev['woman_track_id']} approached by "
                    f"male #{ev['male_track_id']}  "
                    f"dist={ev['distance_px']:.0f}px"
                )

            # Set of affected woman track_ids this frame
            violated_women = {ev["woman_track_id"] for ev in events}
            violated_bboxes = {
                ev["woman_track_id"]: ev["woman_bbox"] for ev in events
            }

            # ── Draw skeletons first (underneath boxes) ──────────────────
            for person in persons:
                pose = pose_map.get(person.track_id)
                if pose is not None:
                    vis = pose_est.draw_skeleton(vis, pose)

            # ── Draw bounding boxes + labels ─────────────────────────────
            for person in persons:
                pose = pose_map.get(person.track_id)
                draw_person_box(vis, person, pose)

                # Flash red X on lone woman if violated this frame
                if person.track_id in violated_women:
                    draw_violation_flash(vis, person.bbox)

            # ── HUD overlay ──────────────────────────────────────────────
            hud_lines = [
                f"FPS: {cam.fps}  |  Persons: {len(persons)}  |  Violations: {violation_total}",
            ]
            if events:
                hud_lines.append(
                    f"[!!] PROXIMITY VIOLATION x{len(events)} this frame"
                )

            for row, line in enumerate(hud_lines):
                colour = ALERT_COLOUR if row > 0 else (0, 255, 255)
                cv2.putText(
                    vis, line, (10, 30 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA,
                )

            cv2.imshow("Sprint 2 — Pose + Proximity Demo", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal proximity violations detected: {violation_total}")
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
