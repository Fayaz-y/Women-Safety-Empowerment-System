import urllib.request
import os

url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4"
filepath = "sample.mp4"

if not os.path.exists(filepath):
    print(f"Downloading {url} to {filepath}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print("Success.")
    except Exception as e:
        print(f"Failed to download: {e}")
        # Try a different one if first fails
        url2 = "https://raw.githubusercontent.com/mamonraab/Real-Time-Violence-Detection-in-Video-/main/hospital.mp4"
        print(f"Trying alternative {url2}...")
        try:
            urllib.request.urlretrieve(url2, filepath)
            print("Success.")
        except Exception as e:
            print(f"Failed again: {e}")

# Run the test
import time
from core.pipeline.engine import PipelineEngine

def on_alert(payload):
    print(f"\n[ALERT TRIGGERED] type: {payload['incident_type']}, score: {payload['fusion_score']:.3f}")
    print(f"Scores -> Violence: {payload['videomae_score']:.3f}, Assault: {payload['bilstm_score']:.3f}, Anomaly: {payload['optflow_score']:.3f}\n")

print("\nStarting PipelineEngine with test video...")
try:
    engine = PipelineEngine(camera_id=99, source=filepath, on_alert=on_alert, device="cuda")
    engine.load_models()
    engine.start()
    
    # Let it run for 15 seconds
    t0 = time.time()
    while time.time() - t0 < 15:
        if not engine.stream.is_open:
            print("Stream closed.")
            break
        fps = engine.stream.fps if hasattr(engine, "stream") else 0
        print(f"Running... FPS: {fps}")
        time.sleep(2)
        
    engine.stop()
    print("Test finished.")
except Exception as e:
    print(f"Error running pipeline: {e}")
