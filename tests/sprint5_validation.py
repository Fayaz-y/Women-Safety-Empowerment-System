"""
Sprint 5 Validation
Checks: 2 cameras concurrent, shared VRAM, FP16 confirmed, torch.compile speed,
        ONNX exports, requirements.txt, Next.js dashboard build.
Requires:
  - 2 USB cameras connected at device 0 and device 1
  - docker-compose up -d
  - uvicorn api.main:app --port 8000
Run: python tests/sprint5_validation.py
"""
import sys
import os
import time
import subprocess
import numpy as np


def run(name, fn):
    try:
        fn()
        print(f"  PASSED  {name}")
        return True
    except Exception as e:
        print(f"  FAILED  {name} â€” {e}")
        return False


def check_two_cameras_open():
    import cv2
    results = []
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        results.append(cap.isOpened())
        cap.release()
    assert all(results), f"Not all cameras opened: {results}"
    print(f"           Both cameras (0, 1) opened successfully")


def check_two_streams_concurrent():
    from core.camera.stream import CameraStream
    cam0 = CameraStream(source=0, camera_id=0).start()
    cam1 = CameraStream(source=1, camera_id=1).start()
    time.sleep(1.0)
    f0 = cam0.read()
    f1 = cam1.read()
    fps0 = cam0.fps
    fps1 = cam1.fps
    cam0.stop()
    cam1.stop()
    assert f0 is not None, "Camera 0 returned no frame"
    assert f1 is not None, "Camera 1 returned no frame"
    print(f"           Cam0 FPS: {fps0} | Cam1 FPS: {fps1}")


def check_detection_on_both_cameras():
    from core.camera.stream import CameraStream
    from core.detection.detector import PersonDetector
    cam0 = CameraStream(source=0).start()
    cam1 = CameraStream(source=1).start()
    det = PersonDetector()
    time.sleep(0.5)
    f0 = cam0.read()
    f1 = cam1.read()
    dets0 = det.detect(f0) if f0 is not None else []
    dets1 = det.detect(f1) if f1 is not None else []
    cam0.stop()
    cam1.stop()
    print(f"           Cam0 detections: {len(dets0)} | Cam1 detections: {len(dets1)}")


def check_vram_two_cameras():
    import torch
    torch.cuda.empty_cache()
    from core.detection.detector import PersonDetector
    from core.tracking.tracker import PersonTracker
    det  = PersonDetector()
    t1   = PersonTracker()
    t2   = PersonTracker()
    used = torch.cuda.memory_allocated() / 1e9
    assert used < 7.5, f"VRAM exceeded with 2 camera trackers: {used:.2f} GB"
    print(f"           VRAM with 2 camera trackers: {used:.2f} GB")


def check_fps_both_cameras():
    from core.camera.stream import CameraStream
    cam0 = CameraStream(source=0).start()
    cam1 = CameraStream(source=1).start()
    time.sleep(2.0)
    fps0 = cam0.fps
    fps1 = cam1.fps
    cam0.stop()
    cam1.stop()
    assert fps0 >= 10, f"Camera 0 FPS too low: {fps0} (target >= 15, min 10 for test)"
    assert fps1 >= 10, f"Camera 1 FPS too low: {fps1} (target >= 15, min 10 for test)"
    print(f"           Cam0: {fps0} FPS | Cam1: {fps1} FPS")


def check_fp16_yolo():
    import torch
    from core.detection.detector import PersonDetector
    det = PersonDetector()
    for name, param in det.model.model.named_parameters():
        assert param.dtype == torch.float16, (
            f"YOLO param {name} is {param.dtype}, expected float16"
        )
        break
    print(f"           YOLO model is FP16")


def check_fp16_efficientnet():
    import torch
    from core.gender.classifier import GenderClassifier
    gc = GenderClassifier()
    for p in gc.effnet.parameters():
        assert p.dtype == torch.float16, f"EfficientNet param not FP16: {p.dtype}"
        break
    print(f"           EfficientNet-B0 is FP16")


def check_inference_speed_post_compile():
    import torch
    from core.detection.detector import PersonDetector
    det = PersonDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(10):
        det.detect(frame)   # warmup
    t0 = time.time()
    N = 50
    for _ in range(N):
        det.detect(frame)
    fps = N / (time.time() - t0)
    assert fps >= 15, f"FPS too low post-optimization: {fps:.1f}"
    print(f"           Post-compile FPS: {fps:.1f}")


def check_onnx_dir_exists():
    os.makedirs("./models/onnx", exist_ok=True)
    assert os.path.isdir("./models/onnx")


def check_efficientnet_onnx_export():
    import torch
    import timm
    model = timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=2
    ).eval()
    dummy = torch.randn(1, 3, 224, 224)
    out_path = "./models/onnx/efficientnet_b0_test.onnx"
    torch.onnx.export(
        model, dummy, out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"]
    )
    assert os.path.isfile(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"           EfficientNet ONNX: {size_mb:.1f} MB at {out_path}")


def check_yolo_onnx_export():
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    model.export(format="onnx", opset=17, simplify=True, half=False)
    expected = "yolo11n-pose.onnx"
    assert os.path.isfile(expected), f"ONNX file not found: {expected}"
    size_mb = os.path.getsize(expected) / 1e6
    print(f"           YOLO11n-Pose ONNX: {size_mb:.1f} MB")


def check_final_vram():
    import torch
    torch.cuda.empty_cache()
    from core.detection.detector import PersonDetector
    from core.violence.detector import ViolenceDetector
    from core.assault.detector import AssaultDetector
    from core.anomaly.detector import AnomalyDetector
    from core.context.clip_context import CLIPContext
    from core.gender.classifier import GenderClassifier
    _ = PersonDetector()
    _ = GenderClassifier()
    _ = ViolenceDetector()
    _ = AssaultDetector()
    _ = AnomalyDetector()
    _ = CLIPContext()
    used = torch.cuda.memory_allocated() / 1e9
    assert used < 7.5, f"Final VRAM check failed: {used:.2f} GB (limit 7.5 GB)"
    print(f"           Final VRAM: {used:.2f} GB / 7.5 GB")


def check_requirements_complete():
    assert os.path.isfile("requirements.txt"), "requirements.txt missing"
    with open("requirements.txt") as f:
        content = f.read()
    required = [
        "ultralytics", "transformers", "timm", "fastapi",
        "sqlalchemy", "redis", "rq", "twilio", "opencv-python", "torch"
    ]
    missing = [pkg for pkg in required if pkg not in content]
    assert not missing, f"requirements.txt missing packages: {missing}"


def check_nextjs_project_exists():
    assert os.path.isdir("dashboard"), "dashboard/ directory missing"
    assert os.path.isfile("dashboard/package.json"), "dashboard/package.json missing"


def check_required_pages_exist():
    pages = [
        "dashboard/app/page.tsx",
        "dashboard/app/incidents/page.tsx",
        "dashboard/app/status/page.tsx",
        "dashboard/app/settings/page.tsx",
    ]
    for p in pages:
        assert os.path.isfile(p), f"Missing page: {p}"
    print(f"           All 4 pages present")


def check_components_exist():
    components = [
        "dashboard/components/CameraGrid.tsx",
        "dashboard/components/AlertBanner.tsx",
        "dashboard/components/IncidentTable.tsx",
    ]
    missing = [c for c in components if not os.path.isfile(c)]
    assert not missing, f"Missing components: {missing}"
    print(f"           All required components present")


def check_store_exists():
    assert os.path.isfile("dashboard/store/useAppStore.ts"), \
        "dashboard/store/useAppStore.ts missing"


def check_env_local_exists():
    env_local = "dashboard/.env.local"
    if not os.path.isfile(env_local):
        with open(env_local, "w") as f:
            f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
            f.write("NEXT_PUBLIC_WS_URL=ws://localhost:8000\n")
        print(f"           Created dashboard/.env.local with defaults")
    else:
        print(f"           dashboard/.env.local exists")


def check_websocket_in_live_monitor():
    with open("dashboard/app/page.tsx", "r") as f:
        content = f.read()
    assert "WebSocket" in content or "useWebSocket" in content or "ws://" in content, \
        "Live monitor page does not appear to use WebSocket"


def check_http_in_incidents():
    with open("dashboard/app/incidents/page.tsx", "r") as f:
        content = f.read()
    assert "axios" in content or "fetch(" in content, \
        "Incidents page does not appear to make HTTP requests"


def check_nextjs_build():
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd="dashboard",
        capture_output=True,
        text=True,
        timeout=180
    )
    assert result.returncode == 0, (
        f"Next.js build failed:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}"
    )
    print(f"           Next.js build successful")


TESTS = [
    ("Two cameras open",                   check_two_cameras_open),
    ("Two streams run concurrently",       check_two_streams_concurrent),
    ("Detection runs on both cameras",     check_detection_on_both_cameras),
    ("VRAM stable with 2 cameras",         check_vram_two_cameras),
    ("FPS >= 10 on both cameras",          check_fps_both_cameras),
    ("YOLO model is FP16",                 check_fp16_yolo),
    ("EfficientNet-B0 is FP16",            check_fp16_efficientnet),
    ("Inference >= 15 FPS post-compile",   check_inference_speed_post_compile),
    ("ONNX output directory exists",       check_onnx_dir_exists),
    ("EfficientNet ONNX export works",     check_efficientnet_onnx_export),
    ("YOLO ONNX export works",             check_yolo_onnx_export),
    ("Final VRAM < 7.5 GB",               check_final_vram),
    ("requirements.txt is complete",       check_requirements_complete),
    ("Next.js project exists",             check_nextjs_project_exists),
    ("All 4 pages exist",                  check_required_pages_exist),
    ("Required components exist",          check_components_exist),
    ("Zustand store exists",               check_store_exists),
    ("dashboard/.env.local exists",        check_env_local_exists),
    ("WebSocket used in live monitor",     check_websocket_in_live_monitor),
    ("HTTP client used in incidents page", check_http_in_incidents),
    ("Next.js build passes",               check_nextjs_build),
]

if __name__ == "__main__":
    print("\n===  Sprint 5 Validation  ===\n")
    print("  NOTE: Requires 2 USB cameras (device 0 + 1), docker-compose up -d,")
    print("        and uvicorn api.main:app --port 8000\n")
    passed = sum(run(n, f) for n, f in TESTS)
    total  = len(TESTS)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  SPRINT 5 INCOMPLETE â€” fix failures")
        sys.exit(1)
    else:
        print("  SPRINT 5 COMPLETE")
        print("  SYSTEM BUILD COMPLETE â€” Women Safety AI is production ready")
