# LeafMedic: Plant Disease Detection — On-Device AI 🪴

[![CI](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml) ![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20it%20now-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blueviolet) ![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-brightgreen) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red) ![Edge AI](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![License](https://img.shields.io/badge/License-MIT-yellowgreen)

**LeafMedic** identifies plant diseases from a photo of a leaf and explains the symptoms, treatment, and prevention for what it finds. A MobileNet neural network classifies **16 conditions across 4 crops** entirely **on-device** — no server, no uploads, no photo ever leaves your machine.

The same network is served three ways: as a **browser app** (ONNX Runtime Web + WebAssembly), as a **desktop app** (Python + PyQt5 with any webcam), and on its original home, a **Raspberry Pi 4** with a camera module. Every diagnosis is paired with treatment and prevention guidance from a built-in knowledge base covering 44 diseases.

---

## 🌐 Live Demo

**➡️ [mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**

> **Note:** The model (~11 MB) downloads once on first visit and is cached afterwards — the page even works offline and can be installed to your phone's home screen.

![plant_interface](https://github.com/user-attachments/assets/b88a5e3d-b727-433a-b9e1-e93642789667)

![raspberry_camera](https://github.com/user-attachments/assets/c243f4a5-1a0e-48db-b00f-42197850fbcb)

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [Supported Crops & Diseases](#-supported-crops--diseases)
- [How It Works](#-how-it-works)
- [Raspberry Pi Setup](#-raspberry-pi-setup)
- [Troubleshooting](#-troubleshooting)
- [Educational Purpose](#-educational-purpose)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## ✨ Features

- 🔍 **Automated disease detection** — ML-powered identification of 16 plant disease classes, with 90%+ confidence on well-captured images
- 💊 **Treatment recommendations** — symptoms, treatment, and prevention guidance for all 44 diseases in the knowledge base
- 🌐 **Zero-install browser demo** — camera, drag-and-drop upload, clipboard paste, or one-click sample images
- 📚 **Disease Library** — browse every supported condition, filterable by crop, with full care guidance
- 🕘 **Recent analyses** — a local history of past diagnoses, stored only in your browser
- 🖥️ **Runs anywhere** — Raspberry Pi + camera module, any laptop with a webcam, or any modern browser
- 🔒 **Private by design** — all inference is on-device; there is no server and no image ever uploads
- ⚡ **Fast inference** — ~30 ms on a laptop, ~145 ms on Raspberry Pi 4; the browser demo runs multi-threaded WebAssembly on repeat visits
- 🛡️ **Reliability guard** — flags images that don't look like a supported leaf instead of returning a confident wrong answer
- 📱 **Installable PWA** — works offline after the first visit (fonts and model included); add it to your phone's home screen
- ✅ **Tested in CI** — golden prediction tests against real sample images plus a headless-browser smoke test on every push

---

## 🛠 Technology Stack

#### Browser Demo
| Technology | Role |
|---|---|
| ONNX Runtime Web (WASM) | In-browser neural network inference |
| Vanilla JS + Canvas API | Camera capture, preprocessing, UI |
| Service Worker | Offline support and asset caching |
| GitHub Pages | Zero-cost static hosting |

#### Desktop App
| Technology | Role |
|---|---|
| Python 3.7+ | Application runtime |
| TensorFlow Lite / LiteRT | Quantized model inference |
| PyQt5 | Graphical interface |
| OpenCV | Webcam capture and image preprocessing |
| Picamera2 | Raspberry Pi camera module support |

#### Model & Data
| Asset | Details |
|---|---|
| MobileNet (uint8-quantized) | 11 MB, 300×300 RGB input, 16 classes |
| [Kaggle AgriPredict](https://www.kaggle.com/models/agripredict/disease-classification) | Pretrained disease-classification model |
| `treatments.json` | Hand-compiled knowledge base, 44 diseases |
| tf2onnx | TFLite → ONNX conversion for the browser (verified identical outputs) |

---

## 🚀 Getting Started

### Prerequisites

- **Browser demo**: any modern browser — nothing else
- **Desktop app**: Python 3.7+, and optionally a webcam
- **Raspberry Pi**: Pi 4 Model B with a camera module (see [Raspberry Pi Setup](#-raspberry-pi-setup))

### Option 1 — Browser (easiest)

Open **[mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**. That's it.

### Option 2 — Desktop app

```bash
git clone https://github.com/mariarodr1136/LeafMedic.git
cd LeafMedic
pip3 install -r requirements.txt
python3 main.py
```

The app auto-detects your camera: it prefers a Raspberry Pi camera module (Picamera2) and falls back to any webcam via OpenCV. No camera at all? Use **Load Image File** — the `test_images/` folder has real samples to try.

### 📸 Capture Tips

| Factor | Recommendation |
|---|---|
| Distance | 20–30 cm from the leaf |
| Lighting | Natural daylight or bright LED; avoid shadows |
| Framing | A single in-focus leaf filling most of the frame |
| Angle | Perpendicular to the leaf surface |
| Background | Plain and contrasting |

---

## 🧭 Development Workflow

| Command | What it does |
|---|---|
| `python3 main.py` | Runs the desktop app |
| `python3 camera_module.py` | Tests camera capture standalone |
| `python3 ml_module.py` | Tests model loading and inference standalone |
| `python3 disease_database.py` | Tests the treatment knowledge base |
| `python3 -m http.server 3000 -d docs` | Serves the browser demo locally at `localhost:3000` |
| `pytest tests/` | Runs data-integrity and golden prediction tests |
| `node tests/web_smoke.mjs` | End-to-end browser demo smoke test (needs `playwright`) |

---

## 📁 Project Structure

```
LeafMedic/
├── main.py                      # Desktop app entry point (run this!)
├── camera_module.py             # Camera control: Picamera2 with OpenCV webcam fallback
├── ml_module.py                 # ML inference with TensorFlow Lite
├── gui_module.py                # PyQt5 graphical interface
├── disease_database.py          # Disease information management
├── download_model.py            # Instructions/helper for obtaining the model
├── requirements.txt             # Python dependencies (desktop)
├── requirements-pi.txt          # Lighter dependencies for Raspberry Pi
├── tests/                       # Pytest suite + browser smoke test (run in CI)
├── .github/workflows/ci.yml    # CI: Python tests + browser smoke test
├── models/
│   ├── plant_disease_model.tflite  # AgriPredict model (11 MB, 16 classes)
│   └── labels.txt               # 16 class labels
├── data/
│   └── treatments.json          # Disease treatment database (44 diseases)
├── test_images/                 # Real sample images to test with
└── docs/                        # Browser demo (GitHub Pages)
    ├── index.html               # Single-page app
    ├── js/inference.js          # ONNX Runtime Web model loading + prediction
    ├── js/app.js                # Camera, upload, results, history, disease library
    ├── model/leafmedic.onnx     # Same network, converted for the browser
    ├── sw.js                    # Service worker (offline + COOP/COEP for threads)
    ├── fonts/                   # Self-hosted Nunito & Outfit (offline-safe)
    ├── img/                     # Hero illustration + disease-cause card images
    └── vendor/                  # ONNX Runtime Web (self-hosted, no CDN)
```

---

## 🧩 Core Components

| Component | Role |
|---|---|
| `main.py` | Wires camera + model + database together and launches the GUI |
| `camera_module.py` | Unified capture API over Picamera2 (Pi) and OpenCV (webcam) |
| `ml_module.py` | TFLite interpreter loading, preprocessing, top-N prediction |
| `disease_database.py` | Loads `treatments.json`, returns care guidance per diagnosis |
| `gui_module.py` | PyQt5 window: live preview, capture, results, image loading |
| `docs/js/inference.js` | Browser twin of `ml_module.py` using ONNX Runtime Web |
| `docs/js/app.js` | Browser UI: input modes, results, history, disease library |

---

## 🌿 Supported Crops & Diseases

![plant_dataset](https://github.com/user-attachments/assets/d8b5b407-6ce5-4f6e-a7d3-cbe942a3c1dd)

| Crop | Classes | Conditions |
|---|---|---|
| 🍅 Tomato | 8 | Healthy · Bacterial Spot · Septoria Leaf Spot · Late Blight ⚠️ · Leaf Mold · Spider Mites · Yellow Leaf Curl Virus ⚠️ |
| 🌽 Corn (maize) | 4 | Healthy · Common Rust · Gray Leaf Spot · Lethal Necrosis ⚠️ |
| 🌱 Soybean | 3 | Healthy · Frogeye Leaf Spot · Downy Mildew |
| 🥬 Cabbage | 2 | Healthy · Black Rot |

> ⚠️ = critical severity. **Total**: 16 classes across 4 crops. The treatment database covers **44 diseases** — the extras (Apple, Grape, Potato, Strawberry, and more) are ready for future model expansion trained on the PlantVillage dataset.

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

1. **Capture** — camera frame or loaded image (RGB)
2. **Preprocess** — resize to 300×300, keep uint8 (0–255); this model needs **no normalization**
3. **Inference** — MobileNet classifies into 16 classes
4. **Post-process** — uint8 outputs → probabilities, 30% confidence floor, take top 3
5. **Display** — diagnosis + confidence, then symptoms/treatment/prevention from the knowledge base

---

## 🔧 Raspberry Pi Setup

The original LeafMedic hardware target — still fully supported.

**Hardware:** Raspberry Pi 4 Model B (4 GB recommended) · Arducam 5 MP OV5647 camera module · monitor · 5V 3A USB-C supply

<details>
<summary><b>Camera connection steps</b></summary>

1. Power off the Raspberry Pi
2. Locate the camera connector (between HDMI ports and audio jack)
3. Pull up the plastic clip gently
4. Insert ribbon cable (**blue side facing audio jack**, contacts facing HDMI)
5. Push the clip down to secure
6. Power on

</details>

```bash
# Enable the camera
sudo raspi-config   # Interface Options → Camera → Enable

# Install dependencies (lightweight LiteRT runtime instead of full TensorFlow)
sudo apt update && sudo apt upgrade -y
sudo apt install python3-picamera2
pip3 install -r requirements-pi.txt --break-system-packages

# Run
python3 main.py
```

---

## 🔍 Troubleshooting

<details>
<summary><b>Camera issues</b></summary>

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

</details>

<details>
<summary><b>ML model issues</b></summary>

**"No TFLite interpreter found"**
```bash
pip3 install tensorflow --break-system-packages
# or the lightweight runtime:
pip3 install ai-edge-litert --break-system-packages
```

**Low confidence predictions (<50%)**
- Verify the plant is a supported crop: **Tomato, Corn, Soybean, or Cabbage only** — anything else produces meaningless low-confidence output
- Use good lighting and a single in-focus leaf filling the frame

</details>

<details>
<summary><b>Browser demo issues</b></summary>

**Camera doesn't start on the web page**
- Allow camera access when the browser asks (the page needs HTTPS, which GitHub Pages provides)
- On iPhone/iPad use Safari; on Android use Chrome
- No camera permission? Upload a photo or use the built-in samples instead

**Model download is slow**
- The 11 MB model + runtime download once and are cached (including offline) afterwards

</details>

<details>
<summary><b>Display issues (desktop app)</b></summary>

**"cannot connect to X server"** — run on the Pi desktop or via VNC, not headless SSH (`export DISPLAY=:0` if needed).

</details>

---

## 📚 Educational Purpose

This project demonstrates:

- **Edge AI deployment** — the same network served three ways (Pi, desktop, browser/WASM)
- **Computer vision** — with OpenCV and Canvas APIs
- **ML inference** — with TensorFlow Lite and ONNX Runtime
- **GUI development** — PyQt5 and a hand-built responsive web UI
- **Modular architecture** — mirrored Python and JavaScript module design

> ⚠️ **Disclaimer:** This is a learning project, not a professional diagnostic tool. The model covers 4 crops / 16 classes — unsupported plants yield meaningless predictions, and visually similar diseases can be confused. For real crop management decisions, consult agricultural extension services or plant pathologists.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

Ideas that would be especially welcome: PlantVillage-trained models with more crops (38+ classes), better treatment recommendations, UI improvements in either app, and bug fixes.

---

## 📄 License

**MIT** — see [LICENSE](LICENSE). Built as an educational project; not a professional diagnostic tool.

- **Model**: Kaggle AgriPredict Disease Classification (16 classes)
- **Dataset (future expansion)**: PlantVillage — CC0 Public Domain
- **Treatment data**: compiled from public agricultural resources
- **Dependencies**: TensorFlow (Apache 2.0) · ONNX Runtime (MIT) · PyQt5 (GPL v3) · OpenCV (Apache 2.0) · Nunito & Outfit fonts (OFL 1.1)

---

## 👏 Acknowledgments

- **AgriPredict** — Disease Classification TFLite model on Kaggle
- **PlantVillage** — 54,000+ image dataset for future model training
- **TensorFlow** & **ONNX Runtime** — inference runtimes
- **Raspberry Pi Foundation** — hardware platform and Picamera2
- **SunFounder** — the electronics kit where this project began

---

## 🌐 Contact

If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).

*Educational Project — Learn, Experiment, Innovate* 🌱
