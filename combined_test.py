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
print("Loading TensorFlow models...")
try:
    gender_model = load_model(r'D:\KPR_HachXelerate\gender\gender_detection.keras')
    gesture_model = tf.keras.models.load_model(r'D:\KPR_HachXelerate\hand_gesture\gesture_recognition_model.h5')
    scaler = load(r'D:\KPR_HachXelerate\hand_gesture\gesture_scaler.joblib')
    print("✅ TensorFlow models loaded successfully")
except Exception as e:
    print(f"❌ Error loading TensorFlow models: {e}")
    exit()

# MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

gesture_map = {0: 'Help', 1: 'Normal'}
gender_classes = ['man', 'woman']

# ---------------- PyTorch Model ----------------
print("Loading PyTorch model...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    violence_model = ViolenceDetector().to(device)
    violence_model.load_state_dict(torch.load("violence_detector.pth", map_location=device))
    violence_model.eval()
    print("✅ PyTorch model loaded successfully")
except Exception as e:
    print(f"❌ Error loading PyTorch model: {e}")
    exit()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
frame_window = []
MAX_FRAMES = 16

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

# ---------------- System Variables ----------------
# Counter variables
help_gesture_count = 0
violence_count = 0
alert_sent = False

# Cooldown timers
last_help_time = 0           # Last time help gesture was detected
last_help_reduce_time = 0    # Last time help count was reduced
last_violence_time = 0       # Last time violence was detected

# Constants for cooldowns (in seconds)
HELP_GESTURE_COOLDOWN = 2.0      # Time between consecutive help gesture detections
HELP_REDUCE_COOLDOWN = 5.0       # Time before reducing help gesture count
VIOLENCE_COOLDOWN = 2.0          # Time between consecutive violence count increments

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

print("✅ System started: Enhanced Gender, Gesture, and Violence Detection")

try:
    while cap.isOpened() and not alert_sent:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        orig_frame = frame.copy()  # Keep a clean copy for violence detection

        # -------- Gender Detection --------
        face, _ = cv.detect_face(frame)
        women_detected = False
        gender_label = "No face"
        
        for f in face:
            (startX, startY), (endX, endY) = f[0:2], f[2:4]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
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
            
            label = f"{gender_label}: {gender_conf[gender_idx]*100:.2f}%"
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------- Hand Gesture Detection --------
        landmarks, hand_landmarks = extract_hand_landmarks(frame)
        gesture = "No hand"
        current_time = time.time()
        
        if landmarks is not None:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_class, gesture_confidence = predict_gesture(landmarks)
            gesture = gesture_map.get(gesture_class, 'Unknown')
            
            if gesture == "Help":
                # Add cooldown for help gesture detection (2 seconds)
                if current_time - last_help_time >= HELP_GESTURE_COOLDOWN:
                    help_gesture_count += 1
                    last_help_time = current_time
                    print(f"🙋‍♀️ Help gesture detected ({help_gesture_count}/5)")
            else:
                # Only reduce help count after 5 seconds of non-help gestures
                if current_time - last_help_reduce_time >= HELP_REDUCE_COOLDOWN:
                    if help_gesture_count > 0:
                        help_gesture_count -= 1
                        print(f"Help gesture count reduced to {help_gesture_count}")
                    last_help_reduce_time = current_time
                
            cv2.putText(frame, f"Gesture: {gesture} ({gesture_confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # -------- Violence Detection --------
        frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(frame_rgb)
        frame_window.append(tensor_frame)
        
        if len(frame_window) > MAX_FRAMES:
            frame_window.pop(0)
            
        violence_detected = False
        violence_pred = "Initializing..."
        
        if len(frame_window) == MAX_FRAMES:
            input_tensor = torch.stack(frame_window).unsqueeze(0).to(device)
            with torch.no_grad():
                output = violence_model(input_tensor)
                is_violence = output.item() > 0.5
                violence_pred = "VIOLENCE" if is_violence else "SAFE"
                violence_color = (0, 0, 255) if is_violence else (0, 255, 0)
                
                # If violence is detected, increment the counter with 2-second cooldown
                if is_violence:
                    violence_detected = True
                    if current_time - last_violence_time >= VIOLENCE_COOLDOWN:
                        violence_count += 1
                        last_violence_time = current_time
                        print(f"🔴 Violence detected! Count: {violence_count}/3")
                
                cv2.putText(frame, f"{violence_pred} ({violence_count}/3)", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, violence_color, 3)

        # -------- Status Display --------
        cooldown_help = max(0, round(HELP_REDUCE_COOLDOWN - (current_time - last_help_reduce_time), 1))
        cooldown_violence = max(0, round(VIOLENCE_COOLDOWN - (current_time - last_violence_time), 1))
        
        status = f"Gender: {gender_label} | Help count: {help_gesture_count}/5 | Violence: {violence_count}/3"
        cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Cooldown indicators
        cooldown_text = f"Help CD: {cooldown_help}s | Violence CD: {cooldown_violence}s"
        cv2.putText(frame, cooldown_text, (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -------- Alert Trigger Logic --------
        alert_reason = None
        
        # Condition 1: Woman detected with help gesture count >= 5
        if women_detected and help_gesture_count >= 5:
            alert_reason = "Woman in distress detected!"
        
        # Condition 2: Violence count reaches 3
        if violence_count >= 3:
            alert_reason = "Repeated violence detected!"
            
        # Send alert if any condition is met
        if alert_reason and not alert_sent:
            print(f"🚨 ALERT: {alert_reason}")
            alert_text = "🚨 ALERT TRIGGERED!"
            cv2.putText(frame, alert_text, (frame.shape[1]//4, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, alert_reason, (frame.shape[1]//4, frame.shape[0]//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Enhanced Detection System", frame)
            cv2.waitKey(1000)
            
            # Send SOS message
            print("📱 Sending SOS message...")
            subprocess.Popen(["python", "code_sms.py"])
            alert_sent = True
            break

        cv2.imshow("Enhanced Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 System shutdown")