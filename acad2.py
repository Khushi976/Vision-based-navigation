import cv2
import torch
import pyttsx3
import os
import numpy as np
from ultralytics import YOLO
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Create output directory if it doesn't exist
output_dir = 'vision_navigation_output'
os.makedirs(output_dir, exist_ok=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()

def give_distance_feedback(objects_with_distances):
    """
    Provide audio feedback for objects closer than 2 meters
    """
    if not objects_with_distances:
        return
    
    for obj_type, distance in objects_with_distances:
        # Convert distance to meters (assuming depth map is in meters)
        distance_meters = round(distance, 5)
        
        if distance_meters < 5:
            # Speak the object type and its distance
            engine.say(f"{obj_type} is {distance_meters} meters ahead.")
    
    engine.runAndWait()

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Load MiDaS model for depth estimation
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Open webcam
cap = cv2.VideoCapture(0)

# Frame counter for saving images
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    try:
        # Object detection
        results = model(frame)
        
        # Annotate the frame with detection results
        annotated_frame = results[0].plot()
        
        # Depth estimation
        inputs = feature_extractor(images=frame, return_tensors="pt")
        with torch.no_grad():
            depth_map = depth_model(**inputs).predicted_depth
        
        # Normalize depth map
        depth_map = depth_map.squeeze().numpy()
        
        # Collect objects with their distances
        objects_with_distances = []
        
        # Process each detected object
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get object class
                cls = model.names[int(box.cls[0])]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get depth at the center of the object
                try:
                    # Ensure center coordinates are within depth map bounds
                    center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                    center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                    
                    # Get distance for this object
                    distance = depth_map[center_y, center_x]
                    
                    objects_with_distances.append((cls, distance))
                except Exception as e:
                    print(f"Error getting distance for {cls}: {e}")
        
        # Provide audio feedback for close objects
        give_distance_feedback(objects_with_distances)
        
        # Save frames and depth maps
        frame_count += 1
        
        # Save original frame
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), frame)
        
        # Save annotated frame with detections
        cv2.imwrite(os.path.join(output_dir, f'detected_{frame_count:04d}.jpg'), annotated_frame)
        
        # Visualize depth map
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, f'depth_{frame_count:04d}.jpg'), depth_vis)
        
        # Print detection and distance information
        print("Detected Objects with Distances:")
        for obj, dist in objects_with_distances:
            print(f"{obj}: {dist:.2f} meters")
        
        # Limit to 100 frames to prevent infinite capture
        if frame_count >= 10:
            break
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()

print(f"Captured {frame_count} frames. Check the '{output_dir}' directory for outputs.")