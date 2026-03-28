"""
Sprint 1 Validation
Checks: CUDA, YOLO11 detection, BoT-SORT tracker, GenderClassifier,
        CameraStream, Settings, FPS, VRAM.
Run: python tests/sprint1_validation.py
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


def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"           Device: {name} | VRAM: {vram:.1f} GB")


def check_yolo_loads():
    from ultralytics import YOLO
    m = YOLO("yolo11n.pt")
    assert m is not None


def check_inference_speed():
    import torch
    from core.detection.detector import PersonDetector
    det = PersonDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        det.detect(frame)
    t0 = time.time()
    N = 30
    for _ in range(N):
        det.detect(frame)
    fps = N / (time.time() - t0)
    assert fps >= 15, f"FPS too low: {fps:.1f} (need >= 15)"
    print(f"           FPS: {fps:.1f}")


def check_vram_yolo_only():
    import torch
    torch.cuda.empty_cache()
    from core.detection.detector import PersonDetector
    det = PersonDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    det.detect(frame)
    used = torch.cuda.memory_allocated() / 1e9
    assert used < 2.0, f"YOLO VRAM too high: {used:.2f} GB"
    print(f"           VRAM used: {used:.2f} GB")


def check_camera_opens():
    import cv2
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot open camera index 0"
    ret, frame = cap.read()
    assert ret and frame is not None, "Frame read failed"
    cap.release()
    print(f"           Frame shape: {frame.shape}")


def check_detection_returns_list():
    from core.detection.detector import PersonDetector
    det = PersonDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = det.detect(frame)
    assert isinstance(result, list)


def check_camera_stream_class():
    from core.camera.stream import CameraStream
    cam = CameraStream(source=0)
    cam.start()
    time.sleep(0.5)
    frame = cam.read()
    cam.stop()
    assert frame is not None, "CameraStream.read() returned None"
    assert len(frame.shape) == 3


def check_settings_loads():
    from config.settings import settings
    assert settings.device in ("cuda", "cpu")
    assert settings.fusion_threshold > 0


def check_tracker_returns_list():
    from core.tracking.tracker import PersonTracker
    tracker = PersonTracker()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = tracker.update(frame)
    assert isinstance(result, list)


def check_tracker_state_dict():
    from core.tracking.tracker import PersonTracker
    tracker = PersonTracker()
    assert isinstance(tracker.state, dict)


def check_tracked_person_fields():
    from core.tracking.tracker import TrackedPerson
    p = TrackedPerson(track_id=1, bbox=[0, 0, 100, 200])
    assert p.gender == "unknown"
    assert p.is_lone_woman is False
    assert p.isolation_seconds == 0.0
    assert isinstance(p.velocity, list)


def check_gender_classifier_loads():
    from core.gender.classifier import GenderClassifier
    gc = GenderClassifier()
    assert gc.effnet is not None
    assert gc.clip_model is not None


def check_gender_output_format():
    from core.gender.classifier import GenderClassifier
    gc = GenderClassifier()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    gender, conf = gc.classify(frame, [100, 100, 300, 500], track_id=1)
    assert gender in ("male", "female", "unknown"), f"Bad gender: {gender}"
    assert 0.0 <= conf <= 1.0, f"Bad conf: {conf}"
    print(f"           gender={gender} conf={conf:.3f}")


def check_gender_cache():
    from core.gender.classifier import GenderClassifier
    gc = GenderClassifier()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    bbox = [50, 50, 200, 400]
    t0 = time.time()
    gc.classify(frame, bbox, track_id=5)
    first_ms = (time.time() - t0) * 1000
    t0 = time.time()
    gc.classify(frame, bbox, track_id=5)
    second_ms = (time.time() - t0) * 1000
    assert second_ms < first_ms, "Cache not working â€” second call should be faster"
    print(f"           First: {first_ms:.1f}ms  Cached: {second_ms:.1f}ms")


def check_vram_sprint1_full():
    import torch
    torch.cuda.empty_cache()
    from core.tracking.tracker import PersonTracker
    from core.gender.classifier import GenderClassifier
    _ = PersonTracker()
    _ = GenderClassifier()
    used = torch.cuda.memory_allocated() / 1e9
    assert used < 3.5, f"VRAM too high after S1 full models: {used:.2f} GB"
    print(f"           VRAM after all S1 models: {used:.2f} GB")


TESTS = [
    ("CUDA available",                  check_cuda),
    ("YOLO11n loads",                   check_yolo_loads),
    ("Inference >= 15 FPS",             check_inference_speed),
    ("YOLO VRAM < 2 GB",                check_vram_yolo_only),
    ("Camera index 0 opens",            check_camera_opens),
    ("detect() returns list",           check_detection_returns_list),
    ("CameraStream reads frames",       check_camera_stream_class),
    ("Settings loads from .env",        check_settings_loads),
    ("Tracker returns list",            check_tracker_returns_list),
    ("Tracker state is dict",           check_tracker_state_dict),
    ("TrackedPerson default fields",    check_tracked_person_fields),
    ("GenderClassifier loads",          check_gender_classifier_loads),
    ("Gender output format valid",      check_gender_output_format),
    ("Gender cache works",              check_gender_cache),
    ("VRAM < 3.5 GB full S1 models",    check_vram_sprint1_full),
]

if __name__ == "__main__":
    print("\n===  Sprint 1 Validation  ===\n")
    passed = sum(run(n, f) for n, f in TESTS)
    total  = len(TESTS)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  SPRINT 1 INCOMPLETE â€” fix failures before Sprint 2")
        sys.exit(1)
    else:
        print("  SPRINT 1 COMPLETE")
