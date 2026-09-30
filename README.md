# Real-Time Hand Tracking and Landmark Estimation Module

An production-ready, object-oriented **Python** implementation designed for real-time human hand detection and landmark tracking using a live camera feed. This module leverages **Google MediaPipe** for robust spatial coordinate predictions and utilizes **OpenCV** for dynamic image manipulation and visualization.

---

## 🌟 Key Features & Improvements

- **Native OpenCV Rendering Architecture:** Unlike standard implementations that rely on the frequently unstable `mediapipe.solutions.drawing_utils` library, this project implements a completely decoupled, manual drawing pipeline. By explicitly mapping the anatomical connection layout, it provides reliable rendering performance across diverse environments.
- **High-Performance Pipeline:** Process high-frame-rate inputs using MediaPipe's lightweight palm detection models with real-time FPS benchmarking overlays.
- **Modular Component Design:** Implemented as a clean, reusable Python class (`handDetector`), allowing developers to import and embed tracking functionalities seamlessly into downstream applications (e.g., gesture control, virtual whiteboards, or sign language recognition).

---

## 📦 Project Directory Structure

```text
├── HandTrackingModule.py   # Core application script containing the tracker class and testing loop
├── README.md               # Project documentation and setup guide
└── requirements.txt        # Pre-requisite third-party libraries
```

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
Download the source files directly to your local machine by executing:
```bash
git clone https://github.com
cd hand-tracking-module
```

### 2. Install Dependencies
Before executing the script, install the verified open-source libraries required by this project via `pip`:
```bash
pip install -r requirements.txt
```
*(Ensure your `requirements.txt` contains `opencv-python` and `mediapipe`)*

---

## 💻 Usage Instructions

### Run the Standalone Demo
To execute the built-in live tracking demonstration, run the module directly from your terminal:
```bash
python HandTrackingModule.py
```
* **Controls:** Press the **`q`** key at any point to release the video capture stream and close the visualization windows safely.

### Integrating the Module into External Projects
You can utilize the `handDetector` class inside other projects with a simple import statement:

```python
import cv2
from HandTrackingModule import handDetector

# Initialize the capture device and tracking engine
cap = cv2.VideoCapture(0)
tracker = handDetector(maxHands=2, detectionCon=0.5, trackCon=0.5)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Process the image frame and display joints/connections
    frame = cv2.flip(frame, 1)
    frame = tracker.findHands(frame, draw=True)
    
    # Extract structural landmark coordinates 
    landmarks = tracker.findPosition(frame, handNo=0)
    
    if len(landmarks) > 0:
        # Landmark index 8 corresponds to the Index Finger Tip
        print(f"Index Finger Coordinates: X={landmarks[8][1]}, Y={landmarks[8][2]}")
        
    cv2.imshow("External Integration Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🔍 Technical Framework Details
The module isolates individual hand profiles and processes **21 distinct structural landmark points** (joints) in a 3D coordinate space ($X, Y, Z$). The coordinates are subsequently scaled automatically to match your device's exact native window resolution.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
