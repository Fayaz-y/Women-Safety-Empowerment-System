"""
women_safety_alert.py

Requirements:
- OpenCV (cv2)
- numpy
- pip install opencv-python-headless (or opencv-python), numpy

Notes:
- Set environment variables EMAIL_ADDRESS and EMAIL_PASSWORD before running.
  Example (Windows PowerShell):
    $env:EMAIL_ADDRESS="your@gmail.com"
    $env:EMAIL_PASSWORD="your_app_password"

- Place your gender model files and cascade XMLs on disk and update the paths below.
"""

import os
import cv2
import numpy as np
import smtplib
import time
from email.message import EmailMessage
import platform
import tempfile
import sys
import math
import traceback
try:
    import winsound
    HAVE_WINSOUND = True
except Exception:
    HAVE_WINSOUND = False

# ------------------ CONFIG ------------------
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # sender
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # app password / environment var
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_ADDRESS)  # default to same address


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Paths - update if your files are elsewhere
HAAR_BODY_PATH = r"C:\Users\Logambika\OneDrive\Attachments\Desktop\Project2026\Women-Safety-Empowerment-System\women_safety\weights\haarcascade_fullbody.xml"
HAAR_FACE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

GENDER_PROTOTXT = r"women_safety/weights/deploy_gender.prototxt"   # update path
GENDER_MODEL = r"women_safety/weights/gender_net.caffemodel"      # update path

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDERS = ['Male', 'Female']

# Thresholds & parameters
BODY_MIN_SIZE = (60, 120)   # min body box size to consider
SURROUND_DISTANCE_SCALE = 1.4   # multiple of female bounding-box width used as proximity radius
SURROUND_COUNT_THRESHOLD = 2    # number of males within radius to consider "surrounded"
EMAIL_COOLDOWN_SECONDS = 30     # don't spam email more than once per cooldown
SAVE_ALERT_IMAGE = True

# --------------------------------------------

# Safety: require email config
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    print("ERROR: EMAIL_ADDRESS and EMAIL_PASSWORD must be set as environment variables.")
    print("Set them and re-run. (Use an app password for Gmail + 2FA.)")
    sys.exit(1)

# Load cascades / models (with checks)
if not os.path.exists(HAAR_BODY_PATH):
    print(f"ERROR: Body cascade not found at {HAAR_BODY_PATH}")
    sys.exit(1)
body_cascade = cv2.CascadeClassifier(HAAR_BODY_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)

if not os.path.exists(GENDER_PROTOTXT) or not os.path.exists(GENDER_MODEL):
    print("ERROR: Gender model files not found. Check GENDER_PROTOTXT and GENDER_MODEL paths.")
    sys.exit(1)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_MODEL)

# helper: play beep (cross-platform best effort)
def beep_alert():
    try:
        if HAVE_WINSOUND and platform.system() == "Windows":
            winsound.Beep(1000, 700)
        else:
            # fallback: print BEL (may or may not beep)
            print("\a")
    except Exception:
        pass

# helper: send email with image attachment
def send_alert_email(subject, body, attachment_path=None):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            data = f.read()
        # guess mime from extension (jpg assumed)
        fname = os.path.basename(attachment_path)
        msg.add_attachment(data, maintype="image", subtype="jpeg", filename=fname)
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            s.send_message(msg)
        print(f"[email] Sent: {subject}")
    except Exception as e:
        print("[email] Failed to send:", e)
        traceback.print_exc()

# gender prediction helper
def predict_gender(face_img):
    # face_img should be BGR
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    idx = int(preds[0].argmax())
    return GENDERS[idx]

# compute centroid from bbox
def centroid(box):
    x, y, w, h = box
    return (int(x + w/2), int(y + h/2))

# distance
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Main capture loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    last_email_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame; exiting")
            break

        frame_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect bodies
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=BODY_MIN_SIZE)
        persons = []  # list of dicts: {bbox, centroid, gender}

        # For each body, attempt to detect face inside it and predict gender
        for (x, y, w, h) in bodies:
            # crop safely
            x1, y1 = max(0,x), max(0,y)
            x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            body_roi = frame[y1:y2, x1:x2]
            gender_label = "Unknown"

            # attempt face detection inside body ROI
            try:
                roi_gray = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]  # choose first
                    face_img = body_roi[fy:fy+fh, fx:fx+fw]
                    # make sure face_img has content
                    if face_img.size != 0:
                        # safe resize & prediction
                        face_resized = cv2.resize(face_img, (227, 227))
                        gender_label = predict_gender(face_resized)
                        # draw face rect inside frame
                        cv2.rectangle(frame_draw, (x1+fx, y1+fy), (x1+fx+fw, y1+fy+fh), (255,255,0), 2)
                else:
                    # no face: keep Unknown
                    gender_label = "Unknown"
            except Exception as e:
                gender_label = "Unknown"

            c = centroid((x1,y1,x2-x1,y2-y1))
            persons.append({"bbox": (x1,y1,x2-x1,y2-y1), "centroid": c, "gender": gender_label})

        # Draw person boxes and labels
        for p in persons:
            (x,y,w,h) = p["bbox"]
            gender = p["gender"]
            color = (255,0,0) if gender=="Female" else (0,255,0)
            if gender=="Unknown":
                color = (200,200,200)
            cv2.rectangle(frame_draw, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame_draw, gender, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Threat detection logic:
        alerts = []
        # Build lists of female and male persons with centroids
        females = [p for p in persons if p["gender"]=="Female"]
        males = [p for p in persons if p["gender"]=="Male"]

        # 1) Woman alone (only one person and it's female or female present and nobody else)
        if len(females) == 1 and len(persons) == 1:
            alerts.append(("ALONE", "Woman detected alone."))

        # 2) Surrounded: for each female, count males within radius proportional to her width
        for f in females:
            fx, fy, fw, fh = f["bbox"]
            fcent = f["centroid"]
            radius = fw * SURROUND_DISTANCE_SCALE
            nearby_males = [m for m in males if euclid(m["centroid"], fcent) <= radius]
            if len(nearby_males) >= SURROUND_COUNT_THRESHOLD:
                alerts.append(("SURROUNDED", f"Woman surrounded by {len(nearby_males)} males."))
                # Optionally draw circle to show radius & male lines
                cv2.circle(frame_draw, fcent, int(radius), (0,0,255), 2)
                for m in nearby_males:
                    cv2.line(frame_draw, fcent, m["centroid"], (0,0,255), 1)

        # 3) Aggressive approach - if male closes rapidly: we can check inter-frame distances (basic)
        # For simplicity, we skip temporal approach detection in this minimal version.
        # You can add tracking (e.g., SORT) to detect rapid approach.

        # Build final alert text
        if alerts:
            # Select highest priority
            # Prioritize SURROUNDED over ALONE
            if any(a[0]=="SURROUNDED" for a in alerts):
                final_alert = next(a for a in alerts if a[0]=="SURROUNDED")[1]
                alert_type = "SURROUNDED"
            else:
                final_alert = alerts[0][1]
                alert_type = alerts[0][0]

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            image_name = f"alert_{alert_type}_{timestamp}.jpg" if SAVE_ALERT_IMAGE else None

            # save image snapshot
            if SAVE_ALERT_IMAGE and image_name:
                cv2.imwrite(image_name, frame_draw)
                print("[saved image]", image_name)

            # beep
            beep_alert()

            # send email but respect cooldown
            now = time.time()
            if now - last_email_time > EMAIL_COOLDOWN_SECONDS:
                subject = f"WOMEN SAFETY ALERT: {alert_type}"
                body = f"Alert: {final_alert}\nTime: {timestamp}\nPlease check the attached image."
                # send email (blocking)
                try:
                    send_alert_email(subject, body, attachment_path=image_name)
                except Exception as e:
                    print("Error sending alert email:", e)
                last_email_time = now
            else:
                print("Email suppressed due to cooldown.")

            # display alert text on frame
            cv2.putText(frame_draw, final_alert, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
        else:
            cv2.putText(frame_draw, "OK: No threat detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # show frame
        cv2.imshow("Women Safety Automation", frame_draw)

        # quit on q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
