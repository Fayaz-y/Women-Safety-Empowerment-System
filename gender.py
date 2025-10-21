import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import subprocess
from joblib import load
import cvlib as cv
import time  # Added for cooldown tracking

# Load gender detection model
try:
    gender_model = load_model(r'D:\KPR_HachXelerate\gender\gender_detection.keras')
    print("Gender detection model loaded successfully")
except Exception as e:
    print(f"Error loading gender model: {e}")
    exit()

# Load hand gesture model and scaler - FIXED PATHS
try:
    gesture_model = tf.keras.models.load_model(r'D:\KPR_HachXelerate\hand_gesture\gesture_recognition_model.h5')
    scaler = load(r'D:\KPR_HachXelerate\hand_gesture\gesture_scaler.joblib')
    print("Hand gesture model loaded successfully")
except Exception as e:
    print(f"Error loading gesture model: {e}")
    exit()

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Gesture and Gender Labels
gesture_map = {0: 'Help', 1: 'Normal'}
gender_classes = ['man', 'woman']

# Camera setup - use one camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Counters and cooldown tracker
help_gesture_count = 0
alert_sent = False
last_help_time = 0  # Cooldown tracker

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
    print(f"Processing landmarks shape: {landmarks.shape}")
    
    landmarks_scaled = scaler.transform(landmarks.reshape(1, -1))
    prediction = gesture_model.predict(landmarks_scaled)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    
    print(f"Gesture prediction: {prediction}, Class: {class_index}, Confidence: {confidence}")
    
    return class_index, confidence

print("Starting Combined Gender and Gesture Detection System...")

try:
    while cap.isOpened() and not alert_sent:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
            
        frame = cv2.flip(frame, 1)
        
        # PART 1: Gender Detection
        face, confidence = cv.detect_face(frame)
        
        women_detected = False
        gender_label = ""
        
        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            face_crop = np.copy(frame[startY:endY, startX:endX])
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue
                
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
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
        
        # PART 2: Hand Gesture Detection
        landmarks, hand_landmarks = extract_hand_landmarks(frame)
        
        help_gesture_detected = False
        gesture = "No hand detected"
        
        if landmarks is not None and hand_landmarks is not None:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_class, gesture_confidence = predict_gesture(landmarks)
            
            gesture = gesture_map.get(gesture_class, f'Unknown')
            
            if gesture == "Help" and women_detected:
                current_time = time.time()
                if current_time - last_help_time >= 2:  # Cooldown of 2 seconds
                    help_gesture_detected = True
                    help_gesture_count += 1
                    last_help_time = current_time
                    print(f"Help gesture detected! Count: {help_gesture_count}")
            else:
                help_gesture_count = max(0, help_gesture_count - 1)
                
            gesture_text = f'Gesture: {gesture} ({gesture_confidence:.2f})'
            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # PART 3: Combined Logic
        status_text = f"Gender: {gender_label} | Gesture: {gesture} | Help count: {help_gesture_count}/5"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        debug_text = f"Women detected: {women_detected}"
        cv2.putText(frame, debug_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if women_detected and help_gesture_count >= 5 and not alert_sent:
            print("🚨 ALERT: Woman detected with Help gesture! Sending SMS...")
            alert_text = "ALERT TRIGGERED!"
            cv2.putText(frame, alert_text, (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Emergency Detection System", frame)
            cv2.waitKey(1000)
            
            subprocess.Popen(["python", r"D:\KPR_HachXelerate\final\drop4.py"])
            alert_sent = True
            break
        
        cv2.imshow("Emergency Detection System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("System shutdown")
