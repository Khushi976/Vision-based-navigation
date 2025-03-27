import cv2
import torch
import pyttsx3
import os
from ultralytics import YOLO
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

# Suppress warnings from Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize text-to-speech engine
engine = pyttsx3.init()

def give_audio_feedback(direction):
    if direction == "left":
        engine.say("Obstacle on the right, move left.")
    elif direction == "right":
        engine.say("Obstacle on the left, move right.")
    elif direction == "front":
        engine.say("Obstacle ahead, stop.")
    engine.runAndWait()

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Load MiDaS model for depth estimation
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame)
    results.show()
    
    # Depth estimation
    inputs = feature_extractor(images=frame, return_tensors="pt")
    with torch.no_grad():
        depth_map = depth_model(**inputs).predicted_depth
    depth_map = depth_map.squeeze().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Show video feeds
    cv2.imshow("Live Camera Feed", frame)
    cv2.imshow("Depth Estimation", depth_map)

    # Check for obstacles and give audio feedback (Example logic, can be improved)
    if depth_map.mean() < 50:  # Adjust threshold as needed
        give_audio_feedback("front")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Required Packages to Install:
# pip install opencv-python torch torchvision torchaudio pyttsx3 transformers ultralytics
