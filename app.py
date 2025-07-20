# app.py
from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for more accuracy

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run YOLOv8 inference
    results = model(filepath)[0]
    img = results.plot()

    _, img_encoded = cv2.imencode('.jpg', img)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        annotated = results.plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
