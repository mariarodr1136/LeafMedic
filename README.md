# LeafMedic: Plant Disease Detection — On-Device AI 🪴🔎

![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20it%20now-brightgreen) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blueviolet) ![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-green) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![GUI](https://img.shields.io/badge/GUI-PyQt5-brightgreen) ![Inference](https://img.shields.io/badge/Inference-Real--Time-success) ![License](https://img.shields.io/badge/License-Educational-yellowgreen)

**LeafMedic** identifies plant diseases from a photo of a leaf and explains the symptoms, treatment, and prevention for what it finds. It's an end-to-end **edge AI application**: a MobileNet model classifies 16 conditions across 4 crops entirely **on-device** — no photo ever leaves your machine.

It runs in **two ways**:

- 🌐 **In your browser** — nothing to install, works with your phone or laptop camera
- 🖥️ **As a desktop app** — Python + PyQt5, on a Raspberry Pi with a camera module *or* any computer with a webcam

---

## 🌐 Try It Now — No Hardware Required

**➡️ [mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**

Three ways to test it, right on the page:

1. **Sample images** — one click on real diseased-leaf photos, no camera or plant needed
2. **Upload** — drag & drop, browse, or paste a leaf photo
3. **Camera** — point your phone or laptop camera at a leaf

The neural network runs **inside your browser** with ONNX Runtime Web (WebAssembly). The model downloads once (~11 MB), is cached for repeat visits, and every diagnosis happens locally — your photos are never uploaded anywhere. The page also works offline after the first visit and can be installed to your phone's home screen.

---

![plant_interface](https://github.com/user-attachments/assets/b88a5e3d-b727-433a-b9e1-e93642789667)

![raspberry_camera](https://github.com/user-attachments/assets/c243f4a5-1a0e-48db-b00f-42197850fbcb)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Raspberry Pi Setup](#-raspberry-pi-setup)
- [Troubleshooting](#-troubleshooting)
- [Educational Purpose](#-educational-purpose)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## ✨ Features

- **Browser demo**: full diagnosis experience with zero install — camera, upload, or sample images
- **Automated Disease Detection**: ML-powered identification of 16 plant disease classes (90%+ accuracy on well-captured images)
- **Treatment Recommendations**: symptoms, treatment, and prevention guidance for all 44 diseases in the knowledge base
- **Disease Library**: browse every supported condition in the web app, filterable by crop
- **Runs anywhere**: Raspberry Pi + camera module, any laptop with a webcam, or any modern browser
- **Private by design**: all inference is on-device; no server, no image uploads
- **Fast Inference**: ~30 ms on a laptop, ~145 ms on Raspberry Pi 4, real-time in-browser via WebAssembly

---

## 🚀 Quick Start

### Option 1 — Browser (easiest)

Open **[mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**. That's it.

### Option 2 — Desktop app (any computer with a webcam)

```bash
git clone https://github.com/mariarodr1136/LeafMedic.git
cd LeafMedic
pip3 install -r requirements.txt
python3 main.py
```

The app auto-detects your camera: it prefers a Raspberry Pi camera module (Picamera2) and falls back to any regular webcam via OpenCV. No camera at all? Use **Load Image File** to analyze saved photos — the `test_images/` folder has real samples to try.

### Option 3 — Raspberry Pi

See [Raspberry Pi Setup](#-raspberry-pi-setup) below.

### GUI Instructions

1. **Live Preview**: camera preview appears in the left panel
2. **Capture & Analyze**: click to capture and diagnose a leaf
3. **View Results**: diagnosis, confidence, and treatment recommendations
4. **Load Image**: analyze saved images with "Load Image File"

### Optimal Image Capture Tips

- **Distance**: 20–30 cm from leaf
- **Lighting**: natural daylight or bright LED (avoid shadows)
- **Focus**: single leaf, fill most of the frame
- **Angle**: perpendicular to the leaf surface
- **Background**: plain, contrasting background helps

### Command-Line Testing

```bash
python3 camera_module.py     # Test camera
python3 ml_module.py         # Test ML model
python3 disease_database.py  # Test disease database
```

---

## 🌿 Supported Plants & Diseases

### Currently Supported (16 Classes — Kaggle AgriPredict Model)

![plant_dataset](https://github.com/user-attachments/assets/d8b5b407-6ce5-4f6e-a7d3-cbe942a3c1dd)

#### 🍅 Tomato (8 classes)
- ✓ **Healthy**
- ⚠️ **Bacterial Spot** — dark spots with yellow halos (High severity)
- ⚠️ **Septoria Leaf Spot** — small circular spots with gray centers (High severity)
- ⚠️ **Late Blight** — devastating disease, can destroy crops rapidly (Critical severity)
- ⚠️ **Leaf Mold** — yellow spots with fuzzy growth underneath (Medium severity)
- ⚠️ **Spider Mites** — pest damage causing yellow stippling (Medium severity)
- ⚠️ **Yellow Leaf Curl Virus (TYLCV)** — viral disease spread by whiteflies (Critical severity)

#### 🌽 Corn/Maize (4 classes)
- ✓ **Healthy**
- ⚠️ **Common Rust** — reddish-brown pustules on leaves (Medium severity)
- ⚠️ **Gray Leaf Spot (Cercospora)** — long narrow gray lesions (High severity)
- ⚠️ **Lethal Necrosis (MLN)** — devastating viral disease, no cure (Critical severity)

#### 🌱 Soybean (3 classes)
- ✓ **Healthy**
- ⚠️ **Frogeye Leaf Spot** — circular lesions with gray centers (Medium severity)
- ⚠️ **Downy Mildew** — yellow spots with fuzzy growth (Medium severity)

#### 🥬 Cabbage (2 classes)
- ✓ **Healthy**
- ⚠️ **Black Rot** — V-shaped yellow lesions, bacterial disease (High severity)

**Total**: 16 disease classes across **4 crop types**. The treatment database covers **44 diseases** — the extras (Apple, Grape, Potato, Strawberry, and more) are ready for future model expansion trained on the PlantVillage dataset.

---

## 🧠 How It Works

### System Architecture

```
        Desktop app                          Browser demo
┌─────────────────────────┐        ┌───────────────────────────┐
│  Pi camera / webcam     │        │  getUserMedia / upload /  │
│  (Picamera2 or OpenCV)  │        │  bundled samples          │
└───────────┬─────────────┘        └────────────┬──────────────┘
            ▼                                   ▼
┌─────────────────────────┐        ┌───────────────────────────┐
│  Preprocess: 300×300    │        │  Canvas resize: 300×300   │
│  RGB uint8              │        │  RGB uint8                │
└───────────┬─────────────┘        └────────────┬──────────────┘
            ▼                                   ▼
┌─────────────────────────┐        ┌───────────────────────────┐
│  TFLite (MobileNet)     │        │  ONNX Runtime Web (WASM)  │
│  plant_disease_model    │        │  leafmedic.onnx           │
└───────────┬─────────────┘        └────────────┬──────────────┘
            └──────────────┬────────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │  Top-3 predictions + %       │
            │  Treatment knowledge base    │
            │  (treatments.json)           │
            └──────────────────────────────┘
```

The browser model is the **exact same network** as the desktop one — converted from TFLite to ONNX with tf2onnx and verified to produce identical predictions on the test set.

### Processing Pipeline

1. **Capture**: camera frame or loaded image (RGB)
2. **Preprocess**: resize to 300×300, keep uint8 (0–255) — this model needs **no normalization**
3. **Inference**: MobileNet classifies into 16 classes (~30 ms laptop / ~145 ms Pi 4 / real-time in browser)
4. **Post-process**: uint8 outputs → probabilities, filter by 30% confidence threshold, take top 3
5. **Display**: diagnosis + confidence, then symptoms/treatment/prevention from the knowledge base

### Model Details

- **Source**: [Kaggle AgriPredict Disease Classification](https://www.kaggle.com/models/agripredict/disease-classification)
- **Architecture**: MobileNet (lightweight CNN optimized for mobile/edge)
- **Input**: 300×300×3 RGB, uint8, raw pixel values
- **Output**: 16 classes, uint8 probabilities (converted to 0–1)
- **Size**: 11 MB (TFLite and ONNX)
- **Accuracy**: 90%+ confidence on well-captured images of supported crops

---

## 📁 Project Structure

```
LeafMedic/
├── main.py                      # Desktop app entry point (run this!)
├── camera_module.py             # Camera control: Picamera2 with OpenCV webcam fallback
├── ml_module.py                 # ML inference with TensorFlow Lite
├── gui_module.py                # PyQt5 graphical interface
├── disease_database.py          # Disease information management
├── requirements.txt             # Python dependencies
├── models/
│   ├── plant_disease_model.tflite  # AgriPredict model (11 MB, 16 classes)
│   └── labels.txt               # 16 class labels
├── data/
│   └── treatments.json          # Disease treatment database (44 diseases)
├── test_images/                 # Real sample images (corn rust, tomato bacterial spot, tomato leaf mold)
└── docs/                        # Browser demo (GitHub Pages)
    ├── index.html               # Single-page app
    ├── js/inference.js          # ONNX Runtime Web model loading + prediction
    ├── js/app.js                # Camera, upload, results, history, disease library
    ├── model/leafmedic.onnx     # Same network, converted for the browser
    └── vendor/                  # ONNX Runtime Web (self-hosted, no CDN)
```

---

## 🔧 Raspberry Pi Setup

The original LeafMedic hardware target — still fully supported.

### Hardware

- **Raspberry Pi 4 Model B** (4 GB recommended)
- **Camera**: Arducam 5 MP OV5647 Camera Module V1 (or compatible)
- **Display**: monitor for the GUI
- **Power**: 5V 3A USB-C supply

### Camera Connection

1. Power off the Raspberry Pi
2. Locate the camera connector (between HDMI ports and audio jack)
3. Pull up the plastic clip gently
4. Insert ribbon cable (**blue side facing audio jack**, contacts facing HDMI)
5. Push the clip down to secure
6. Power on

### Software

```bash
# Enable the camera
sudo raspi-config   # Interface Options → Camera → Enable

# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-picamera2
pip3 install tensorflow opencv-python --break-system-packages

# Run
python3 main.py
```

---

## 🔍 Troubleshooting

### Camera Issues

**"Camera not available" on Raspberry Pi**
```bash
libcamera-hello --list-cameras   # Should show: ov5647 [2592x1944]
sudo raspi-config                # Enable camera if not detected
sudo reboot
```

**"Camera in use by another process"**
```bash
sudo pkill -9 libcamera
sudo pkill -9 rpicam
```

**No webcam picture on desktop**
- macOS: grant camera permission to your terminal in System Settings → Privacy & Security → Camera
- Close other apps that might hold the camera (Zoom, browser tabs)
- No camera is fine — use "Load Image File" instead

### ML Model Issues

**"No TFLite interpreter found"**
```bash
pip3 install tensorflow --break-system-packages
# or the lightweight runtime:
pip3 install ai-edge-litert --break-system-packages
```

**Low confidence predictions (<50%)**
- Verify the plant is a supported crop: **Tomato, Corn, Soybean, or Cabbage only** — anything else produces meaningless low-confidence output
- Use good lighting, a single in-focus leaf filling the frame
- See "Optimal Image Capture Tips" above

### Browser Demo Issues

**Camera doesn't start on the web page**
- Allow camera access when the browser asks (the page needs HTTPS, which GitHub Pages provides)
- On iPhone/iPad use Safari; on Android use Chrome
- No camera permission? Upload a photo or use the built-in samples instead

**Model download is slow**
- The 11 MB model + 13 MB runtime download once and are cached (including offline) afterwards

### Display Issues (desktop app)

**"cannot connect to X server"** — run on the Pi desktop or via VNC, not headless SSH (`export DISPLAY=:0` if needed).

---

## 📚 Educational Purpose

This project demonstrates:

- Edge AI deployment: the same network served three ways (Pi, desktop, browser/WASM)
- Computer vision with OpenCV and Canvas APIs
- Machine learning inference with TensorFlow Lite and ONNX Runtime
- GUI development with PyQt5 and a hand-built responsive web UI
- Modular Python and JavaScript architecture

### Limitations & Disclaimers

⚠️ **IMPORTANT**:

- This is a **learning project**, not a professional diagnostic tool
- The model covers **4 crops / 16 classes** — unsupported plants yield meaningless predictions
- False positives/negatives occur; visually similar diseases can be confused
- One diagnosis per image (highest confidence wins)
- For real crop management decisions, consult agricultural extension services or plant pathologists

---

## 🤝 Contributing

Contributions welcome:

- Improved models with more crops (PlantVillage-trained, 38+ classes)
- Better treatment recommendations
- UI improvements in either app
- Bug fixes and optimizations

---

## 📄 License

**Educational Use Only**

Provided as-is for learning computer vision and machine learning concepts.

**Model & Data**:
- Kaggle AgriPredict: Disease Classification model (16 classes)
- PlantVillage dataset: CC0 (Public Domain) — future expansion
- Treatment information compiled from public agricultural resources

**Dependencies**: TensorFlow (Apache 2.0), ONNX Runtime (MIT), PyQt5 (GPL v3), OpenCV (Apache 2.0)

---

## 👏 Acknowledgments

- **AgriPredict** — Disease Classification TFLite model on Kaggle
- **PlantVillage** — dataset for future model training (54,000+ images)
- **TensorFlow** & **ONNX Runtime** — inference runtimes
- **Raspberry Pi Foundation** — hardware platform and Picamera2
- **SunFounder** — the electronics kit where this project began

---

## 🌐 Contact
If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).

*Educational Project — Learn, Experiment, Innovate*
