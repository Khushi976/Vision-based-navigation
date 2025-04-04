
# Vision-Based Navigation

This project implements a vision-based navigation system that utilizes object detection and depth estimation to help avoid obstacles in real-time.

## Features

- **Object Detection:** Uses YOLO (from the ultralytics package) to detect objects in the video feed.
- **Depth Estimation:** Employs a MiDaS model via Hugging Face's transformers to compute a depth map.
- **Audio Feedback:** Provides audio feedback (using pyttsx3) based on obstacle detection.

## Requirements

Make sure to install the following packages:
- OpenCV: `pip install opencv-python`
- PyTorch: `pip install torch torchvision torchaudio`
- pyttsx3: `pip install pyttsx3`
- Transformers: `pip install transformers`
- Ultralytics: `pip install ultralytics`

## Usage

1. Ensure your webcam is connected.
2. Run the script:
   ```bash
   python acad2.py
