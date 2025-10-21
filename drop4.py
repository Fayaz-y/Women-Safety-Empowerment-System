import cv2
import dropbox
import time
import serial
from datetime import datetime
import keys
from twilio.rest import Client
import requests
import os
import ssl
import urllib3
from dropbox.dropbox_client import Dropbox
from dropbox import create_session

# Disable SSL warnings (local dev only!)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize Dropbox with SSL check disabled
session = create_session()
session.verify = False
dbx = Dropbox(keys.dropbox_access_token, session=session)

def send_alert_to_server(name, loc):
    try:
        response = requests.post("http://192.168.137.100:5000/alert", json={"name": name, "loc": loc})
        if response.status_code == 200:
            print(f"✅ Alert sent with name: {name}")
            print("The server will redirect any open browser sessions to index1.html")
        else:
            print("❌ Failed to alert server.")
    except Exception as e:
        print(f"❌ Error: {e}")

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"captured_video_{timestamp}.mp4"
video_path = os.path.join("D:/KPR_HachXelerate/web_dashboard/static/video", video_filename)
dropbox_path = f"/videos/{video_filename}"

# Capture video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_size = (640, 480)

out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
start_time = time.time()

while int(time.time() - start_time) < 10:  # Record for 10 seconds
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print("🎥 Video Captured Successfully!")

# Upload to Dropbox
try:
    with open(video_path, "rb") as file:
        dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
    print("✅ File uploaded to Dropbox!")

    loc = "https://www.google.com/maps?q=11.07771,77.14296"
    # Create shareable link
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    dropbox_link = shared_link_metadata.url.replace("?dl=0", "?raw=1")  # Direct viewable link
    print("🔗 Dropbox File Link:", dropbox_link)
     
    send_alert_to_server(dropbox_link, loc)
    # Send SMS with Twilio
    client = Client(keys.account_sid, keys.account_token)
    loc = "https://www.google.com/maps?q=11.07771,77.14296"

    message = client.messages.create(
        from_=keys.twilio_number,
        body=f"""🚨 Emergency alert!
Location: {loc}
Video Evidence: {dropbox_link}""",
        to=keys.my_phone_number
    )

    print(f"📩 SMS sent! Message SID: {message.sid}")
    

except Exception as e:
    print("❌ Failed to upload to Dropbox or send SMS:", e)
