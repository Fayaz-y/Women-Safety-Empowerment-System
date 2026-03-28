"""
Women Safety AI — ONNX Export Script
======================================
Exports all compatible models to ONNX format for future deployment
(Jetson, TensorRT, ONNX Runtime).

Output directory: ./models/onnx/

Usage:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import os
import sys

import torch
import timm
from ultralytics import YOLO

ONNX_DIR = os.path.join(".", "models", "onnx")


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def export_yolo_pose():
    """Export YOLO11n-Pose to ONNX using Ultralytics' built-in exporter."""
    print("\n[1/3] Exporting YOLO11n-Pose ...")
    model = YOLO("yolo11n-pose.pt")
    result = model.export(
        format="onnx",
        dynamic=True,
        half=False,
        opset=17,
        simplify=True,
    )
    # Ultralytics saves alongside the .pt file — move to ONNX dir
    src_path = result if isinstance(result, str) else "yolo11n-pose.onnx"
    if os.path.exists(src_path):
        dest = os.path.join(ONNX_DIR, "yolo11n-pose.onnx")
        if os.path.abspath(src_path) != os.path.abspath(dest):
            import shutil
            shutil.copy2(src_path, dest)
        print(f"  ✓ YOLO11n-Pose → {dest}  ({_file_size_mb(dest):.1f} MB)")
    else:
        # Ultralytics might save next to the .pt
        alt = "yolo11n-pose.onnx"
        if os.path.exists(alt):
            dest = os.path.join(ONNX_DIR, "yolo11n-pose.onnx")
            import shutil
            shutil.copy2(alt, dest)
            print(f"  ✓ YOLO11n-Pose → {dest}  ({_file_size_mb(dest):.1f} MB)")
        else:
            print("  ⚠ YOLO export completed but ONNX file not found")


def export_efficientnet_b0():
    """Export EfficientNet-B0 gender classifier to ONNX."""
    print("\n[2/3] Exporting EfficientNet-B0 (gender) ...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path = os.path.join(ONNX_DIR, "efficientnet_b0_gender.onnx")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"  ✓ EfficientNet-B0 → {out_path}  ({_file_size_mb(out_path):.1f} MB)")


def export_assault_bilstm():
    """Export the VGG19 + BiLSTM assault model to ONNX."""
    print("\n[3/3] Exporting Assault BiLSTM ...")

    # Import the private model architecture
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.assault.detector import _AssaultModel

    model = _AssaultModel()
    model.eval()

    T = 16
    VGG_FEAT_DIM_IMG = (3, 224, 224)
    POSE_DIM = 17 * 3

    dummy_frames = torch.randn(1, T, *VGG_FEAT_DIM_IMG)
    dummy_poses = torch.randn(1, T, POSE_DIM)

    out_path = os.path.join(ONNX_DIR, "assault_bilstm.onnx")

    try:
        torch.onnx.export(
            model,
            (dummy_frames, dummy_poses),
            out_path,
            opset_version=17,
            input_names=["frame_feats", "poses"],
            output_names=["logits"],
            dynamic_axes={
                "frame_feats": {0: "batch"},
                "poses": {0: "batch"},
            },
        )
        print(f"  ✓ Assault BiLSTM → {out_path}  ({_file_size_mb(out_path):.1f} MB)")
    except Exception as exc:
        print(f"  ⚠ Assault BiLSTM export failed: {exc}")
        print("    (BiLSTM + VGG19 exports may require tracing mode)")


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)
    print("=" * 60)
    print("  ONNX Export — Women Safety AI Models")
    print(f"  Output directory: {os.path.abspath(ONNX_DIR)}")
    print("=" * 60)

    export_yolo_pose()
    export_efficientnet_b0()
    export_assault_bilstm()

    print("\n" + "=" * 60)
    print("  Export complete")
    print("=" * 60)

    # List exported files
    for f in os.listdir(ONNX_DIR):
        fp = os.path.join(ONNX_DIR, f)
        if os.path.isfile(fp):
            print(f"  {f}  ({_file_size_mb(fp):.1f} MB)")


if __name__ == "__main__":
    main()
