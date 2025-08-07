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

# --------------------------
# 📷 Open webcam
# --------------------------
cap = cv2.VideoCapture(1)
print("🔴 Starting real-time violence detection...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
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
            cv2.putText(frame, prediction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            if prediction == "VIOLENCE":
                count +=1
            else:
                count -=1

    if count > 3:
        subprocess.Popen(["python", "drop4.py"])
        alert_sent = True
    
        break
    # Display frame
    cv2.imshow("Live Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()