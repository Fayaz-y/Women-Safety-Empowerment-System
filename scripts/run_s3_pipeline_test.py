"""
Women Safety AI — Sprint 3 Pipeline Test Script
=================================================
Verifies the full pipeline runs without crashing.

Usage:
    python scripts/run_s3_pipeline_test.py
"""

import sys
import time
sys.path.insert(0, ".")

from core.pipeline.engine import PipelineEngine


def main():
    """Run the Sprint 3 full pipeline test."""
    print("=" * 60)
    print("  Sprint 3 — Full Pipeline Test")
    print("=" * 60)
    print()

    # Create pipeline engine with alert callback
    engine = PipelineEngine(
        camera_id=1,
        source=0,
        on_alert=lambda p: print(
            f"[ALERT] {p['incident_type']} "
            f"score={p['fusion_score']:.3f}"
        ),
    )

    # Load all models
    engine.load_models()

    # Start the pipeline
    engine.start()

    # Run for 60 seconds, printing FPS every 5 seconds
    print("\nPipeline running for 60 seconds...")
    print("-" * 40)
    try:
        for i in range(12):  # 12 × 5s = 60s
            time.sleep(5)
            fps = engine.stream.fps
            elapsed = (i + 1) * 5
            print(f"[{elapsed:3d}s] FPS: {fps}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        engine.stop()
        print()
        print("=" * 60)
        print("  Pipeline stopped — test complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
