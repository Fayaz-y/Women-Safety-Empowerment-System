import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import subprocess
from joblib import load
import cvlib as cv
import time
import torch
import torchvision.transforms as transforms
from model import ViolenceDetector  # Your PyTorch model class

# ---------------- TensorFlow Models ----------------
gender_model = load_model(r'D:\KPR_HachXelerate\gender\gender_detection.keras')
gesture_model = tf.keras.models.load_model(r'D:\KPR_HachXelerate\hand_gesture\gesture_recognition_model.h5')
scaler = load(r'D:\KPR_HachXelerate\hand_gesture\gesture_scaler.joblib')

# MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

gesture_map = {0: 'Help', 1: 'Normal'}
gender_classes = ['man', 'woman']

# ---------------- PyTorch Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
violence_model = ViolenceDetector().to(device)
violence_model.load_state_dict(torch.load("violence_detector.pth", map_location=device))
violence_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
frame_window = []
MAX_FRAMES = 16

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)
help_gesture_count = 0
alert_sent = False
last_help_time = 0

def extract_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), hand_landmarks
    return None, None

def predict_gesture(landmarks):
    landmarks_scaled = scaler.transform(landmarks.reshape(1, -1))
    prediction = gesture_model.predict(landmarks_scaled)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return class_index, confidence

print("✅ System started: Gender, Gesture, and Violence Detection")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        orig_frame = frame.copy()

        # -------- Gender Detection --------
        face, _ = cv.detect_face(frame)
        women_detected = False
        gender_label = ""
        for f in face:
            (startX, startY), (endX, endY) = f[0:2], f[2:4]
            face_crop = np.copy(frame[startY:endY, startX:endX])
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue
            face_crop = cv2.resize(face_crop, (96, 96)).astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)
            gender_conf = gender_model.predict(face_crop)[0]
            gender_idx = np.argmax(gender_conf)
            gender_label = gender_classes[gender_idx]
            if gender_label == 'woman':
                women_detected = True
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{gender_label}: {gender_conf[gender_idx]*100:.2f}%"
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------- Hand Gesture Detection --------
        landmarks, hand_landmarks = extract_hand_landmarks(frame)
        gesture = "No hand"
        if landmarks is not None:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_class, gesture_confidence = predict_gesture(landmarks)
            gesture = gesture_map.get(gesture_class, 'Unknown')
            if gesture == "Help":
                current_time = time.time()
                if current_time - last_help_time >= 2:
                    help_gesture_count += 1
                    last_help_time = current_time
                    print(f"🙋‍♀️ Help gesture detected ({help_gesture_count})")
            else:
                help_gesture_count = max(0, help_gesture_count - 1)
            cv2.putText(frame, f"Gesture: {gesture} ({gesture_confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # -------- Violence Detection --------
        frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(frame_rgb)
        frame_window.append(tensor_frame)
        if len(frame_window) > MAX_FRAMES:
            frame_window.pop(0)
        if len(frame_window) == MAX_FRAMES:
            input_tensor = torch.stack(frame_window).unsqueeze(0).to(device)
            with torch.no_grad():
                output = violence_model(input_tensor)
                violence_pred = "VIOLENCE" if output.item() > 0.5 else "SAFE"
                violence_color = (0, 0, 255) if violence_pred == "VIOLENCE" else (0, 255, 0)
                cv2.putText(frame, violence_pred, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, violence_color, 3)

        # -------- Alert Trigger --------
        status = f"Gender: {gender_label} | Gesture: {gesture} | Help: {help_gesture_count}/5"
        cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if women_detected and help_gesture_count >= 5 and not alert_sent:
            print("🚨 ALERT: Woman in distress detected!")
            cv2.putText(frame, "🚨 ALERT TRIGGERED!", (frame.shape[1]//4, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Combined Detection", frame)
            cv2.waitKey(1000)
            subprocess.Popen(["python", "code_sms.py"])
            alert_sent = True
            break

        cv2.imshow("Combined Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 System shutdown")
