from flask import Flask, request, render_template, send_file, Response, redirect, url_for
import pytesseract
from PIL import Image
from gtts import gTTS
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
import os
import torch
import subprocess

app = Flask(__name__)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("YOLO model loaded successfully!")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml')

# Load TensorFlow object detection model
detection_model = tf.saved_model.load(
    'models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Variables for object detection cooldown
last_detected = None
last_time = 0
audio_cooldown = 3  # seconds

# Function to detect objects and speak their names using YOLO


# Function to detect objects and speak their names using YOLO
def detect_and_speak(frame):
    global last_detected, last_time
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame captured, skipping detection.")
        return None  # Skip processing if no valid frame is available

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = model(frame_rgb)  # Run detection on the RGB frame

    # Extract predictions from results
    detections = results.xyxy[0]  # Get predictions for the first image

    if len(detections) > 0:
        for detection in detections:
            class_id = int(detection[5])  # Class index
            confidence = detection[4].item()  # Confidence score

            if confidence > 0.5:
                object_name = results.names[class_id]  # Get the class name
                print("Detected:", object_name)

                # Only speak if the detected object is new or cooldown has passed
                if object_name != last_detected or (time.time() - last_time) > audio_cooldown:
                    last_detected = object_name
                    last_time = time.time()

                    # Speak the detected object name
                    tts = gTTS(text=object_name, lang='en')
                    audio_path = "object.mp3"

                    # Wait for any ongoing audio playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    # Save audio and play it
                    tts.save(audio_path)
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                    # Wait for the audio to finish playing before deleting the file
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    # Now it's safe to delete the file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

                results.render()  # Draw results on the frame

    # Convert frame to JPEG for streaming
    return cv2.imencode('.jpg', frame)[1].tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("No frame captured.")
            continue  # Skip the loop iteration if frame capture failed

        frame_with_detections = detect_and_speak(frame)  # Process the frame
        if frame_with_detections is None:
            print("Detection returned None.")
            continue  # Skip if processing failed

        print("Sending frame to video feed.")  # Debug line
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_detections + b'\r\n')

# Camera class for managing camera


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video device")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success or frame is None:
            print("Failed to capture a valid frame inside Flask app")
            return None
        return frame


@app.route('/run_detection')
def run_detection():
    try:
        subprocess.Popen(['python', 'try.py'])
    except Exception as e:
        return str(e)

    return redirect(url_for('camera_page'))

# Home route


@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload (if needed)


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']

        # Process image (e.g., OCR for text recognition)
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)

        # Generate audio from text using gTTS
        tts = gTTS(text=text, lang='en')
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)  # Reset buffer to start

        # Send audio file back as a response
        return send_file(audio_io, mimetype='audio/mpeg')

    # If it's a GET request, render the upload page
    return render_template('upload.html')

# Video feed route


@app.route('/video_feed')
def video_feed():
    camera = Camera()
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

# Camera page route


@app.route('/camera')
def camera_page():
    return render_template('camera.html')


if __name__ == "__main__":
    app.run(debug=True)
