import cv2
import numpy as np
import onnxruntime
from datetime import datetime
import json
import os

class WasteDetector:
    def __init__(self, model_path='models/waste_model.onnx', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.initialize_model(model_path)
        self.cap = cv2.VideoCapture(0)
        self.logs = []
        self.classes = ['plastic', 'metal', 'glass', 'paper', 'organic', 'e-waste', 'mixed']
        self.colors = {
            'plastic': (255, 0, 0),    # Blue
            'metal': (0, 255, 0),      # Green
            'glass': (0, 0, 255),      # Red
            'paper': (255, 255, 0),    # Cyan
            'organic': (255, 0, 255),  # Magenta
            'e-waste': (0, 255, 255),  # Yellow
            'mixed': (128, 128, 128)   # Gray
        }
        self.is_running = False

    def initialize_model(self, model_path):
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_image(self, frame):
        input_height, input_width = self.input_shape[2:]
        img = cv2.resize(frame, (input_width, input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def process_frame(self):
        if not self.is_running:
            ret, frame = self.cap.read()
            if ret:
                return frame, []
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # Preprocess the frame
        input_tensor = self.preprocess_image(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process detections
        detections = self.process_detections(outputs[0], frame)
        
        # Draw detections on frame
        self.draw_detections(frame, detections)
        
        return frame, detections

    def process_detections(self, output, frame):
        detections = []
        
        # Process the model output to get bounding boxes, classes and confidences
        for detection in output[0]:
            confidence = detection[4]
            if confidence < self.conf_threshold:
                continue
                
            class_id = int(detection[5])
            if class_id >= len(self.classes):
                continue
                
            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            x1 = int(detection[0] * w)
            y1 = int(detection[1] * h)
            x2 = int(detection[2] * w)
            y2 = int(detection[3] * h)
            
            detection_info = {
                'class': self.classes[class_id],
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            }
            detections.append(detection_info)
            
            # Log the detection
            self.log_detection(detection_info)
            
        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def log_detection(self, detection):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'waste_type': detection['class'],
            'confidence': detection['confidence'],
            'bbox': detection['bbox']
        }
        self.logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(self.logs) > 1000:
            self.logs.pop(0)

    def get_logs(self):
        return self.logs

    def start_detection(self):
        self.is_running = True

    def stop_detection(self):
        self.is_running = False

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
