"""
Women Safety AI — Pipeline Orchestration Engine
=================================================
One instance per camera. Loads all models, starts the camera stream,
runs the full detection pipeline in a background thread, and calls
`on_alert` when threats are detected.

Key flow changes (v2):
  • Threat assessment runs on ALL tracked females every ASSESS_EVERY
    frames — NOT gated behind proximity events
  • Proximity events, surrounded, and lone-woman-night are ADDITIONAL
    signals that boost the fusion score
  • Rich video annotations via FrameAnnotator
  • Immediate alert dispatch on any fusion trigger

Usage:
    engine = PipelineEngine(camera_id=1, source=0, on_alert=handle_alert)
    engine.load_models()
    engine.start()
    engine.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional, Union

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
from core.pipeline.annotator import FrameAnnotator
from core.pipeline.clip_saver import ClipSaver


class PipelineEngine:
    """
    Main orchestration engine — one instance per camera.

    Runs the full detection → assessment → fusion pipeline in a daemon
    thread and fires on_alert callbacks immediately when threats are detected.
    """

    def __init__(
        self,
        camera_id: int,
        source: Union[int, str],
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
        
        # Video clip saver
        self.clip_saver = ClipSaver(clip_dir=settings.clip_dir)

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load all models sequentially."""
        def _vram_gb() -> float:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
            return 0.0

        print(f"\n{'='*60}")
        print(f"  PipelineEngine — Loading models (camera {self.camera_id})")
        print(f"{'='*60}\n")

        # 1. Camera stream
        print(f"[1/11] CameraStream(source={self.source}) ...")
        self.stream = CameraStream(self.source, self.camera_id).start()
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 2. Person tracker
        print("[2/11] PersonTracker ...")
        self.tracker = PersonTracker(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 3. Gender classifier
        print("[3/11] GenderClassifier ...")
        self.gender_clf = GenderClassifier(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 4. Pose estimator
        print("[4/11] PoseEstimator ...")
        self.pose_est = PoseEstimator(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 5. Proximity engine
        print("[5/11] ProximityEngine ...")
        self.proximity = ProximityEngine()
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 6. Violence detector (or shared)
        if "violence" in self.shared_models:
            print("[6/11] ViolenceDetector (shared) ...")
            self.violence = self.shared_models["violence"]
        else:
            print("[6/11] ViolenceDetector ...")
            self.violence = ViolenceDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 7. Assault detector (or shared)
        if "assault" in self.shared_models:
            print("[7/11] AssaultDetector (shared) ...")
            self.assault = self.shared_models["assault"]
        else:
            print("[7/11] AssaultDetector ...")
            self.assault = AssaultDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 8. Anomaly detector (or shared)
        if "anomaly" in self.shared_models:
            print("[8/11] AnomalyDetector (shared) ...")
            self.anomaly = self.shared_models["anomaly"]
        else:
            print("[8/11] AnomalyDetector ...")
            self.anomaly = AnomalyDetector(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 9. CLIP context (or shared)
        if "clip" in self.shared_models:
            print("[9/11] CLIPContext (shared) ...")
            self.clip_ctx = self.shared_models["clip"]
        else:
            print("[9/11] CLIPContext ...")
            self.clip_ctx = CLIPContext(device=self.device)
        print(f"       ✓ VRAM: {_vram_gb():.2f} GB")

        # 10. Fusion layer
        print("[10/11] FusionLayer ...")
        # Use configurable threshold from settings (default 0.75, lower for more sensitivity)
        from config.settings import settings
        self.fusion = FusionLayer(threshold=settings.fusion_threshold)
        print(f"        ✓ VRAM: {_vram_gb():.2f} GB")

        # 11. Frame annotator
        print("[11/11] FrameAnnotator ...")
        self.annotator = FrameAnnotator(camera_name=f"Camera {self.camera_id}")
        print(f"        ✓ Ready")

        # VRAM summary
        vram = _vram_gb()
        print(f"\n  Total VRAM: {vram:.2f} GB")
        if torch.cuda.is_available() and vram > 7.5:
            print(f"  ⚠ WARNING: VRAM usage {vram:.2f} GB exceeds 7.5 GB budget")
        print(f"  ✓ All models loaded\n")

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

        # Per-track threat scores for annotation
        threat_scores: Dict[int, float] = {}

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
                pr = pose_map.get(person.track_id)
                kp = pr.keypoints if pr else None

                # Classify gender (with keypoints for body proportions)
                gender, conf = self.gender_clf.classify(
                    frame, person.bbox, person.track_id, keypoints=kp
                )
                person.gender = gender
                person.gender_conf = conf

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

            # ── Proximity + Surrounded + Night ─────────────────────────────
            events = self.proximity.update(persons, frame)

            # Get current status sets for annotation
            lone_ids = self.proximity.get_lone_woman_ids(persons)
            surrounded_ids = self.proximity.get_surrounded_ids(persons)

            # ── Anomaly (every frame) ─────────────────────────────────────
            anomaly_result = self.anomaly.compute(frame)

            # ── CLIP context (every 30 frames, cache result) ──────────────
            if frame_count % 30 == 0:
                clip_result = self.clip_ctx.score(frame)
                _last_clip_score = clip_result["clip_threat_score"]

            # ── Threat assessment — on ALL tracked females ──────────────
            if frame_count % ASSESS_EVERY == 0:
                females = [p for p in persons if p.gender == "female"]

                for woman in females:
                    tid = woman.track_id
                    v_result = self.violence.predict(tid)
                    a_result = self.assault.predict(tid)
                    pr = pose_map.get(tid)

                    # Base scores from models
                    videomae_s = v_result["violence_score"] if v_result else 0.0
                    bilstm_s = a_result["assault_confidence"] if a_result else 0.0
                    optflow_s = anomaly_result["anomaly_score"]
                    clip_s = _last_clip_score
                    pose_s = pr.distress_score if pr else 0.0

                    # Boost scores based on situational context
                    context_boost = 0.0
                    incident_type = "threat_detected"

                    # Check if this woman has proximity events
                    woman_events = [
                        e for e in events if e.get("woman_track_id") == tid
                    ]
                    for evt in woman_events:
                        if evt["type"] == "proximity_violation":
                            context_boost += 0.15
                            incident_type = "proximity_threat"
                        elif evt["type"] == "woman_surrounded":
                            context_boost += 0.25
                            incident_type = "woman_surrounded"
                        elif evt["type"] == "lone_woman_night":
                            context_boost += 0.10
                            incident_type = "lone_woman_night"

                    # If surrounded, boost the score significantly
                    if tid in surrounded_ids:
                        context_boost = max(context_boost, 0.20)

                    inp = FusionInput(
                        videomae_score=videomae_s,
                        bilstm_score=bilstm_s,
                        optflow_score=optflow_s,
                        clip_score=clip_s,
                        pose_score=pose_s,
                    )

                    fusion_result = self.fusion.compute(inp)

                    # Apply context boost
                    boosted_score = min(
                        fusion_result.fusion_score + context_boost, 1.0
                    )

                    # Store threat score for annotation
                    threat_scores[tid] = boosted_score

                    # Check if alert should fire
                    if (fusion_result.triggered or
                            boosted_score >= settings.fusion_threshold):
                        # Determine specific incident type
                        if a_result and a_result.get("is_assault"):
                            incident_type = "assault_detected"
                        elif v_result and v_result.get("is_violent"):
                            incident_type = "violence_detected"

                        self._dispatch_alert(
                            tid, incident_type, boosted_score,
                            fusion_result, frame, frame_count,
                        )

            # ── Annotate frame ─────────────────────────────────────────────
            annotated = self.annotator.annotate(
                frame=frame,
                persons=persons,
                threat_scores=threat_scores,
                lone_woman_ids=lone_ids,
                surrounded_ids=surrounded_ids,
                night_mode=self.proximity.is_night,
                fps=self.stream.fps,
                frame_count=frame_count,
            )

            # ── Buffer frame for clip saving ──────────────────────────────
            self.clip_saver.add_frame(annotated)

            # ── Store annotated frame (thread-safe) ───────────────────────
            with self._lock:
                self._last_annotated = annotated

    # ── Alert dispatch ────────────────────────────────────────────────────────

    def _dispatch_alert(
        self,
        woman_track_id: int,
        incident_type: str,
        boosted_score: float,
        fusion_result,
        frame: np.ndarray,
        frame_count: int,
    ) -> None:
        """Fire on_alert callback with cooldown suppression."""
        now = time.time()
        last_alert = self._alert_cooldowns.get(woman_track_id, 0)

        if (now - last_alert) < settings.alert_cooldown_seconds:
            return  # suppress duplicate

        self._alert_cooldowns[woman_track_id] = now

        # Flash alert on annotator
        self.annotator.set_alert_fired(frame_count)

        print(
            f"  [ALERT] cam{self.camera_id} | {incident_type} | "
            f"score={boosted_score:.3f} | track={woman_track_id}"
        )

        if self.on_alert is not None:
            self.on_alert({
                "camera_id":      self.camera_id,
                "incident_type":  incident_type,
                "fusion_score":   boosted_score,
                "videomae_score": fusion_result.input_scores.videomae_score,
                "bilstm_score":   fusion_result.input_scores.bilstm_score,
                "optflow_score":  fusion_result.input_scores.optflow_score,
                "clip_score":     fusion_result.input_scores.clip_score,
                "pose_score":     fusion_result.input_scores.pose_score,
                "woman_track_id": woman_track_id,
                "timestamp":      now,
                "frame":          frame,
                "clip_path":      self._save_incident_clip(
                    incident_type, woman_track_id, boosted_score
                ),
            })

    # ── Annotated frame access ────────────────────────────────────────────────

    def _save_incident_clip(
        self, incident_type: str, woman_track_id: int, fusion_score: float
    ) -> Optional[str]:
        """Save the buffered frames as an MP4 clip with incident labels."""
        return self.clip_saver.save_clip(
            incident_type=incident_type,
            camera_id=self.camera_id,
            fusion_score=fusion_score,
            woman_track_id=woman_track_id,
        )

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest annotated frame (thread-safe)."""
        with self._lock:
            if self._last_annotated is not None:
                return self._last_annotated.copy()
            return None
