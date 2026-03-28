"""
Women Safety AI — Pipeline Orchestration Engine
=================================================
One instance per camera. Loads all models, starts the camera stream,
runs the full detection pipeline in a background thread, and calls
`on_alert` when the fusion threshold is exceeded.

Usage:
    def handle_alert(payload):
        print(f"ALERT: {payload['incident_type']} score={payload['fusion_score']}")

    engine = PipelineEngine(camera_id=1, source=0, on_alert=handle_alert)
    engine.load_models()
    engine.start()
    # ... engine runs in background ...
    engine.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from config.settings import settings
from core.camera.stream import CameraStream
from core.tracking.tracker import PersonTracker
from core.gender.classifier import GenderClassifier
from core.pose.estimator import PoseEstimator
from core.proximity.engine import ProximityEngine
from core.violence.detector import ViolenceDetector
from core.assault.detector import AssaultDetector
from core.anomaly.detector import AnomalyDetector
from core.context.clip_context import CLIPContext
from core.fusion.fusion import FusionLayer, FusionInput


class PipelineEngine:
    """
    Main orchestration engine — one instance per camera.

    Loads all Sprint 1–3 models, runs the full detection → assessment →
    fusion pipeline in a daemon thread, and fires on_alert callbacks.
    """

    def __init__(
        self,
        camera_id: int,
        source: int,
        on_alert: Optional[Callable[[dict], None]] = None,
        device: str = "cuda",
        shared_models: dict = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.on_alert = on_alert
        self.device = device
        self.shared_models = shared_models or {}

        self.running = False
        self._last_annotated: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._alert_cooldowns: Dict[int, float] = {}

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """
        Load all models sequentially with VRAM checks.

        Models can be shared via shared_models dict (keys: violence,
        assault, anomaly, clip) for multi-camera VRAM sharing.
        """
        def _vram_gb() -> float:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
            return 0.0

        print(f"\n{'='*60}")
        print(f"  PipelineEngine — Loading models (camera {self.camera_id})")
        print(f"{'='*60}\n")

        # 1. Camera stream
        print(f"[1/10] CameraStream(source={self.source}) ...")
        self.stream = CameraStream(self.source, self.camera_id).start()
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 2. Person tracker
        print("[2/10] PersonTracker ...")
        self.tracker = PersonTracker(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 3. Gender classifier
        print("[3/10] GenderClassifier ...")
        self.gender_clf = GenderClassifier(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 4. Pose estimator
        print("[4/10] PoseEstimator ...")
        self.pose_est = PoseEstimator(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 5. Proximity engine
        print("[5/10] ProximityEngine ...")
        self.proximity = ProximityEngine()
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 6. Violence detector (or shared)
        if "violence" in self.shared_models:
            print("[6/10] ViolenceDetector (shared) ...")
            self.violence = self.shared_models["violence"]
        else:
            print("[6/10] ViolenceDetector ...")
            self.violence = ViolenceDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 7. Assault detector (or shared)
        if "assault" in self.shared_models:
            print("[7/10] AssaultDetector (shared) ...")
            self.assault = self.shared_models["assault"]
        else:
            print("[7/10] AssaultDetector ...")
            self.assault = AssaultDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 8. Anomaly detector (or shared)
        if "anomaly" in self.shared_models:
            print("[8/10] AnomalyDetector (shared) ...")
            self.anomaly = self.shared_models["anomaly"]
        else:
            print("[8/10] AnomalyDetector ...")
            self.anomaly = AnomalyDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 9. CLIP context (or shared)
        if "clip" in self.shared_models:
            print("[9/10] CLIPContext (shared) ...")
            self.clip_ctx = self.shared_models["clip"]
        else:
            print("[9/10] CLIPContext ...")
            self.clip_ctx = CLIPContext(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 10. Fusion layer
        print("[10/10] FusionLayer ...")
        self.fusion = FusionLayer()
        print(f"        ✓ VRAM: {_vram_gb():.2f} GB")

        # VRAM assertion
        vram = _vram_gb()
        print(f"\n  Total VRAM: {vram:.2f} GB")
        if torch.cuda.is_available():
            assert vram < 7.5, (
                f"VRAM budget exceeded! Using {vram:.2f} GB (limit 7.5 GB)"
            )
        print(f"  ✓ All models loaded — VRAM within budget\n")

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the pipeline loop in a daemon thread."""
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"[PipelineEngine] Started (camera {self.camera_id})")

    def stop(self) -> None:
        """Stop the pipeline and release the camera."""
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop()
        print(f"[PipelineEngine] Stopped (camera {self.camera_id})")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Internal pipeline loop — runs in a background daemon thread."""
        ASSESS_EVERY = 8   # run full threat assessment every 8 frames
        frame_count = 0
        _last_clip_score = 0.0

        while self.running:
            frame = self.stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            annotated = frame.copy()

            # ── Tracking ──────────────────────────────────────────────────
            persons = self.tracker.update(frame)

            # ── Pose estimation ───────────────────────────────────────────
            poses = self.pose_est.estimate(frame)
            pose_map = {pr.track_id: pr for pr in poses}

            # ── Per-person: gender + frame buffer push + skeleton draw ────
            for person in persons:
                gender, conf = self.gender_clf.classify(
                    frame, person.bbox, person.track_id
                )
                person.gender = gender
                person.gender_conf = conf

                pr = pose_map.get(person.track_id)
                if pr:
                    # Crop person using person.bbox
                    x1 = max(0, int(person.bbox[0]))
                    y1 = max(0, int(person.bbox[1]))
                    x2 = int(person.bbox[2])
                    y2 = int(person.bbox[3])
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # Push to rolling frame buffers for violence + assault
                        self.violence.push_frame(person.track_id, crop)
                        self.assault.push_frame(
                            person.track_id, crop, pr.keypoints
                        )

                    # Draw skeleton
                    annotated = self.pose_est.draw_skeleton(annotated, pr)

            # ── Proximity ─────────────────────────────────────────────────
            events = self.proximity.update(persons)

            # ── Anomaly (every frame) ─────────────────────────────────────
            anomaly_result = self.anomaly.compute(frame)

            # ── CLIP context (every 30 frames, cache result) ──────────────
            if frame_count % 30 == 0:
                clip_result = self.clip_ctx.score(frame)
                _last_clip_score = clip_result["clip_threat_score"]

            # ── Threat assessment (on proximity violations) ───────────────
            if events and frame_count % ASSESS_EVERY == 0:
                for event in events:
                    tid = event["woman_track_id"]
                    v_result = self.violence.predict(tid)
                    a_result = self.assault.predict(tid)
                    pr = pose_map.get(tid)

                    inp = FusionInput(
                        videomae_score=(
                            v_result["violence_score"] if v_result else 0.0
                        ),
                        bilstm_score=(
                            a_result["assault_confidence"] if a_result else 0.0
                        ),
                        optflow_score=anomaly_result["anomaly_score"],
                        clip_score=_last_clip_score,
                        pose_score=pr.distress_score if pr else 0.0,
                    )

                    fusion_result = self.fusion.compute(inp)

                    if fusion_result.triggered:
                        self._dispatch_alert(event, fusion_result, frame)

            # ── Store annotated frame (thread-safe) ───────────────────────
            with self._lock:
                self._last_annotated = annotated

    # ── Alert dispatch ────────────────────────────────────────────────────────

    def _dispatch_alert(
        self,
        event: dict,
        fusion_result,
        frame: np.ndarray,
    ) -> None:
        """Fire on_alert callback with cooldown suppression."""
        tid = event["woman_track_id"]
        now = time.time()
        last_alert = self._alert_cooldowns.get(tid, 0)

        if (now - last_alert) < settings.alert_cooldown_seconds:
            return  # suppress duplicate

        self._alert_cooldowns[tid] = now

        if self.on_alert is not None:
            self.on_alert({
                "camera_id":      self.camera_id,
                "incident_type":  "proximity_threat",
                "fusion_score":   fusion_result.fusion_score,
                "videomae_score": fusion_result.input_scores.videomae_score,
                "bilstm_score":   fusion_result.input_scores.bilstm_score,
                "optflow_score":  fusion_result.input_scores.optflow_score,
                "clip_score":     fusion_result.input_scores.clip_score,
                "pose_score":     fusion_result.input_scores.pose_score,
                "woman_track_id": event["woman_track_id"],
                "timestamp":      event["timestamp"],
                "frame":          frame,
            })

    # ── Annotated frame access ────────────────────────────────────────────────

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest annotated frame (thread-safe)."""
        with self._lock:
            if self._last_annotated is not None:
                return self._last_annotated.copy()
            return None
