import time
import os
import torch
import cv2
import numpy as np

# A loop to robustly download VGG19 if needed
def ensure_vgg19():
    import urllib.request
    checkpoint = os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth")
    url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    if not os.path.exists(checkpoint) or os.path.getsize(checkpoint) < 500000000:
        print(f"Downloading VGG19 to {checkpoint}...")
        for i in range(5):
            try:
                urllib.request.urlretrieve(url, checkpoint)
                print("Downloaded VGG19.")
                return
            except Exception as e:
                print(f"Retry {i+1} failed: {e}")
                time.sleep(2)
        print("Failed to download VGG19.")

ensure_vgg19()

from core.camera.stream import CameraStream
from core.tracking.tracker import PersonTracker
from core.pose.estimator import PoseEstimator
from core.assault.detector import AssaultDetector

print("Models importing done.")

video_path = "real_assault_test.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("Loading Tracker...")
    tracker = PersonTracker(device=device)
    print("Loading Pose Estimator...")
    pose_est = PoseEstimator(device=device)
    print("Loading Assault Detector...")
    assault = AssaultDetector(device=device)
    print("All models loaded!")
    
    cap = cv2.VideoCapture(video_path)
    
    # Setup VideoWriter
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_assault_test.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
            
        frame_count += 1
        persons = tracker.update(frame)
        poses = pose_est.estimate(frame)
        for pr in poses:
            xs = [k[0] for k in pr.keypoints if k[2] > 0.1]
            ys = [k[1] for k in pr.keypoints if k[2] > 0.1]
            if not xs or not ys: 
                continue
            x1, y1, x2, y2 = min(xs)-30, min(ys)-50, max(xs)+30, max(ys)+30
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = int(x2), int(y2)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                assault.push_frame(pr.track_id, crop, pr.keypoints)
                res = assault.predict(pr.track_id)
                if res:
                    conf = res['assault_confidence']
                    is_assault = res['is_assault']
                    print(f"Frame {frame_count} - PoseTrack {pr.track_id}: conf={conf:.3f}, is_assault={is_assault}")
                    
                    # Annotate frame
                    color = (0, 0, 255) if conf > 0.50 else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"Track {pr.track_id} | Assault: {conf:.2f}"
                    cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        out_video.write(frame)
        
    cap.release()
    out_video.release()
    print("Targeted test finished. Output saved to 'output_assault_test.mp4'.")

except Exception as e:
    import traceback
    traceback.print_exc()
