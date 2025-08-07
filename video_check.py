import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from model import ViolenceDetector
import subprocess

# --------------------------
# 🧠 Load your trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViolenceDetector().to(device)
model.load_state_dict(torch.load("violence_detector.pth", map_location=device))
model.eval()

# --------------------------
# 📦 Frame preprocessor
# --------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

frame_window = []
MAX_FRAMES = 16

count = 0

# --------------------------
# 📼 Open video file
# --------------------------
video_path = r"D:\KPR_HachXelerate\Violence\Dataset\Violence\V_105.mp4"  # 🔁 Replace with your actual file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

print("🎬 Starting violence detection on video...")

# 📺 Create a resizable display window
cv2.namedWindow("Video Violence Detection", cv2.WINDOW_NORMAL)
# Optional: Resize window to a reasonable default (comment this if you want full resolution)
cv2.resizeWindow("Video Violence Detection", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(frame_rgb)
    frame_window.append(tensor_frame)

    if len(frame_window) > MAX_FRAMES:
        frame_window.pop(0)

    # Run prediction
    if len(frame_window) == MAX_FRAMES:
        input_tensor = torch.stack(frame_window).unsqueeze(0).to(device)  # [1, T, C, H, W]
        with torch.no_grad():
            output = model(input_tensor)
            prediction = "VIOLENCE" if output.item() > 0.5 else "NON-VIOLENCE"
            color = (0, 0, 255) if prediction == "VIOLENCE" else (0, 255, 0)
            if prediction == "VIOLENCE" :
                count+=1
            else:
                count+=1

    # Keep original frame size for display
    display_frame = frame.copy()

    if len(frame_window) == MAX_FRAMES:
        cv2.putText(display_frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if count > 8:
        subprocess.Popen(["python", "drop4.py"])
        alert_sent = True
    # Show result
    cv2.imshow("Video Violence Detection", display_frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
