from flask import Flask, request, render_template, send_file, Response, redirect, url_for
import pytesseract
from PIL import Image
from gtts import gTTS
from io import BytesIO
import cv2
import time
import os
import pygame
import subprocess
import uuid

from ultralytics import YOLO

app = Flask(__name__)

# Initialize pygame mixer at the start
pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.mixer.init()

# YOLO model setup 
model = YOLO('yolov8n.pt')

# Variables for object detection cooldown
last_detected = None
last_time = 0
audio_cooldown = 3  # seconds


def delete_file_safely(file_path, retries=5, delay=0.5):
    """Try to delete a file safely with retries."""
    for _ in range(retries):
        try:
            if os.path.exists(file_path):
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                time.sleep(0.5)

                os.remove(file_path)
                print(f"Deleted audio file: {file_path}")
            return
        except PermissionError:
            print(
                f"PermissionError: Unable to delete {file_path}. Retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    print(f"Failed to delete {file_path} after multiple attempts.")


def detect_and_speak(frame):
    global last_detected, last_time

    current_time = time.time()

    # Cooldown check
    if last_detected and current_time - last_time < audio_cooldown:
        return cv2.imencode('.jpg', frame)[1].tobytes()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    detections = []
    if results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if confidence > 0.5:
                detections.append(
                    {'label': model.names[class_id], 'confidence': confidence})

    if detections:
        detected_objects = ", ".join(
            [f"{obj['label']} ({obj['confidence']:.2f})" for obj in detections])
        text_to_speak = f"Detected: {detected_objects}"
    else:
        text_to_speak = "No objects detected."

    audio_path = f"object_{uuid.uuid4().hex}.mp3"
    last_detected = detections
    last_time = current_time

    try:
        delete_file_safely(audio_path)

        tts = gTTS(text=text_to_speak, lang='en')
        tts.save(audio_path)
        print(f"Audio file saved: {audio_path}")

        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        print("Audio playback finished.")
    except Exception as e:
        print(f"Error during audio playback: {e}")

    if hasattr(results, 'plot'):
        frame = results[0].plot()

    return cv2.imencode('.jpg', frame)[1].tobytes()


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Replace 0 if needed
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open video device")

    def get_frame(self):
        success, frame = self.cap.read()
        if not success or frame is None:
            print("Retrying to capture frame...")
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            success, frame = self.cap.read()
            if not success or frame is None:
                print("Still unable to capture frame.")
                return None
        return frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def gen(camera):
    """Generator function for video streaming."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_with_detections = detect_and_speak(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_detections + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/camera')
def camera_page():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    camera = Camera()
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        tts = gTTS(text=text, lang='en')
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return send_file(audio_io, mimetype='audio/mpeg')
    return render_template('upload.html')


@app.route('/run_detection')
def run_detection():
    try:
        subprocess.Popen(['python', 'try.py'])
    except Exception as e:
        return str(e)
    return redirect(url_for('camera_page'))


if __name__ == "__main__":
    app.run(debug=True)
