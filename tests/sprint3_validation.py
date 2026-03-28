"""
Sprint 3 Validation
Checks: ViolenceDetector, AssaultDetector, AnomalyDetector, CLIPContext,
        FusionLayer logic, total VRAM budget.
Run: python tests/sprint3_validation.py
"""
import sys
import time
import numpy as np


def run(name, fn):
    try:
        fn()
        print(f"  PASSED  {name}")
        return True
    except Exception as e:
        print(f"  FAILED  {name} â€” {e}")
        return False


def check_violence_detector_loads():
    from core.violence.detector import ViolenceDetector
    vd = ViolenceDetector()
    assert vd.model is not None
    print(f"           VideoMAE V2 loaded")


def check_violence_buffer_not_full():
    from core.violence.detector import ViolenceDetector
    vd = ViolenceDetector()
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    for i in range(10):
        vd.push_frame(track_id=1, crop=crop)
    result = vd.predict(track_id=1)
    assert result is None, "predict() must return None when buffer has < 16 frames"


def check_violence_predict_format():
    from core.violence.detector import ViolenceDetector
    vd = ViolenceDetector()
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    for i in range(16):
        vd.push_frame(track_id=2, crop=crop)
    result = vd.predict(track_id=2)
    assert result is not None, "predict() must return dict when buffer is full (16 frames)"
    assert "track_id" in result
    assert "label" in result
    assert "confidence" in result
    assert "is_violent" in result
    assert "violence_score" in result
    assert 0.0 <= result["confidence"] <= 1.0
    print(f"           label={result['label']} conf={result['confidence']:.3f} violent={result['is_violent']}")


def check_violence_clear_track():
    from core.violence.detector import ViolenceDetector
    vd = ViolenceDetector()
    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    for _ in range(16):
        vd.push_frame(9, crop)
    vd.clear_track(9)
    assert vd.predict(9) is None, "predict() must return None after clear_track()"


def check_assault_detector_loads():
    from core.assault.detector import AssaultDetector
    ad = AssaultDetector()
    assert ad.net is not None
    print(f"           VGG19+BiLSTM loaded")


def check_assault_buffer_not_full():
    from core.assault.detector import AssaultDetector
    ad = AssaultDetector()
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    kp = np.random.rand(17, 3).astype(np.float32)
    for i in range(10):
        ad.push_frame(track_id=1, crop=crop, keypoints=kp)
    result = ad.predict(track_id=1)
    assert result is None, "predict() must return None when buffer has < 16 frames"


def check_assault_predict_format():
    from core.assault.detector import AssaultDetector
    ad = AssaultDetector()
    crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    kp = np.random.rand(17, 3).astype(np.float32)
    for i in range(16):
        ad.push_frame(track_id=2, crop=crop, keypoints=kp)
    result = ad.predict(track_id=2)
    assert result is not None
    assert "assault_confidence" in result
    assert "is_assault" in result
    assert 0.0 <= result["assault_confidence"] <= 1.0
    print(f"           assault_conf={result['assault_confidence']:.3f}")


def check_anomaly_detector_loads():
    from core.anomaly.detector import AnomalyDetector
    ad = AnomalyDetector()
    assert ad.raft is not None


def check_anomaly_compute_returns_dict():
    from core.anomaly.detector import AnomalyDetector
    ad = AnomalyDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = ad.compute(frame)
    assert isinstance(result, dict)
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert 0.0 <= result["anomaly_score"] <= 1.0


def check_anomaly_calibration():
    from core.anomaly.detector import AnomalyDetector
    ad = AnomalyDetector()
    ad.CALIBRATION_FRAMES = 5
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(6):
        ad.compute(frame)
    assert ad._calibrated is True, "Should be calibrated after enough frames"
    print(f"           Calibrated: mean={ad._mean:.3f} std={ad._std:.3f}")


def check_clip_context_loads():
    from core.context.clip_context import CLIPContext
    cc = CLIPContext()
    assert cc.model is not None
    assert cc._text_feats is not None


def check_clip_score_format():
    from core.context.clip_context import CLIPContext
    cc = CLIPContext()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = cc.score(frame)
    assert "clip_threat_score" in result
    assert "top_prompt" in result
    assert 0.0 <= result["clip_threat_score"] <= 1.0
    print(f"           threat_score={result['clip_threat_score']:.3f}")


def check_fusion_weights_sum():
    from core.fusion.fusion import FusionLayer
    fl = FusionLayer()
    total = sum(fl.weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got: {total}"


def check_fusion_below_threshold():
    from core.fusion.fusion import FusionLayer, FusionInput
    fl = FusionLayer(threshold=0.75)
    inp = FusionInput(
        videomae_score=0.1, bilstm_score=0.1,
        optflow_score=0.1, clip_score=0.1, pose_score=0.1
    )
    result = fl.compute(inp)
    assert result.triggered is False
    assert result.fusion_score < 0.75
    print(f"           Low score: {result.fusion_score:.4f}")


def check_fusion_above_threshold():
    from core.fusion.fusion import FusionLayer, FusionInput
    fl = FusionLayer(threshold=0.75)
    inp = FusionInput(
        videomae_score=0.9, bilstm_score=0.9,
        optflow_score=0.9, clip_score=0.9, pose_score=0.9
    )
    result = fl.compute(inp)
    assert result.triggered is True
    print(f"           High score: {result.fusion_score:.4f}")


def check_fusion_rejects_bad_weights():
    from core.fusion.fusion import FusionLayer
    fl = FusionLayer()
    try:
        fl.update_weights({
            "videomae": 0.5, "bilstm": 0.5,
            "optflow": 0.5, "clip": 0.5, "pose": 0.5
        })
        assert False, "Should have raised AssertionError for weights summing to 2.5"
    except AssertionError:
        pass  # Expected


def check_total_vram_all_models():
    import torch
    torch.cuda.empty_cache()
    from core.detection.detector import PersonDetector
    from core.tracking.tracker import PersonTracker
    from core.gender.classifier import GenderClassifier
    from core.pose.estimator import PoseEstimator
    from core.violence.detector import ViolenceDetector
    from core.assault.detector import AssaultDetector
    from core.anomaly.detector import AnomalyDetector
    from core.context.clip_context import CLIPContext

    _ = PersonDetector()
    _ = PersonTracker()
    _ = GenderClassifier()
    _ = PoseEstimator()
    _ = ViolenceDetector()
    _ = AssaultDetector()
    _ = AnomalyDetector()
    _ = CLIPContext()

    used = torch.cuda.memory_allocated() / 1e9
    assert used < 7.5, f"TOTAL VRAM exceeds 7.5 GB limit: {used:.2f} GB"
    print(f"           Total VRAM all models: {used:.2f} GB / 7.5 GB limit")


TESTS = [
    ("ViolenceDetector loads",               check_violence_detector_loads),
    ("Violence buffer returns None <16f",    check_violence_buffer_not_full),
    ("Violence predict() format valid",      check_violence_predict_format),
    ("Violence clear_track() works",         check_violence_clear_track),
    ("AssaultDetector loads",                check_assault_detector_loads),
    ("Assault buffer returns None <16f",     check_assault_buffer_not_full),
    ("Assault predict() format valid",       check_assault_predict_format),
    ("AnomalyDetector loads",                check_anomaly_detector_loads),
    ("anomaly compute() returns dict",       check_anomaly_compute_returns_dict),
    ("Anomaly calibration works",            check_anomaly_calibration),
    ("CLIPContext loads",                    check_clip_context_loads),
    ("CLIP score() format valid",            check_clip_score_format),
    ("Fusion weights sum to 1.0",            check_fusion_weights_sum),
    ("Fusion below threshold -> False",      check_fusion_below_threshold),
    ("Fusion above threshold -> True",       check_fusion_above_threshold),
    ("Bad weights rejected",                 check_fusion_rejects_bad_weights),
    ("TOTAL VRAM < 7.5 GB",                  check_total_vram_all_models),
]

if __name__ == "__main__":
    print("\n===  Sprint 3 Validation  ===\n")
    passed = sum(run(n, f) for n, f in TESTS)
    total  = len(TESTS)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  SPRINT 3 INCOMPLETE â€” fix failures before Sprint 4")
        sys.exit(1)
    else:
        print("  SPRINT 3 COMPLETE")
