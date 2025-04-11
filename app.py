from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import time
from detection import WasteDetector
import json
import csv
import io
from datetime import datetime

app = Flask(__name__)

# Initialize the waste detector
detector = WasteDetector()

def generate_frames():
    while True:
        frame, detections = detector.process_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)  # Small delay to prevent overwhelming the system

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    return render_template('logs.html', logs=detector.get_logs())

@app.route('/export')
def export():
    return render_template('export.html')

@app.route('/export/json')
def export_json():
    logs = detector.get_logs()
    return jsonify(logs)

@app.route('/export/csv')
def export_csv():
    logs = detector.get_logs()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['timestamp', 'waste_type', 'confidence', 'bbox'])
    writer.writeheader()
    writer.writerows(logs)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    output.close()
    
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'waste_detection_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, port=8000)
