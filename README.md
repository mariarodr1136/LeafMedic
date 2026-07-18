# LeafMedic: Plant Disease Detection — On-Device AI

[![CI](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml) ![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20it%20now-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blueviolet) ![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-brightgreen) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red) ![Edge AI](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![License](https://img.shields.io/badge/License-MIT-yellowgreen)

**LeafMedic** identifies plant diseases from a photo of a leaf and explains the symptoms, treatment, and prevention for what it finds. A fully integer-quantized MobileNet CNN classifies **16 conditions across 4 crops** entirely **on-device** — no server, no uploads, no photo ever leaves your machine.

The same network is deployed through three runtimes from a single source of truth: a **browser app** (ONNX Runtime Web on WebAssembly), a **desktop app** (Python + PyQt5 with any webcam), and its original target, a **Raspberry Pi 4** with a camera module. Every diagnosis is paired with treatment and prevention guidance from a structured knowledge base covering 44 diseases.

It is built as a study in **edge AI deployment constraints**: an 11 MB quantized graph converted across two inference runtimes with verified prediction parity, sub-150 ms CPU inference on a Raspberry Pi, multi-threaded WebAssembly unlocked on a static host that cannot set HTTP headers, and an out-of-distribution guard so the classifier declines to guess instead of confidently misdiagnosing an unsupported plant.

---

## Live Demo

**[mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**

> **Note:** The model (~11 MB) downloads once on first visit and is cached afterwards — the page even works offline and can be installed to your phone's home screen.

![plant_interface](https://github.com/user-attachments/assets/b88a5e3d-b727-433a-b9e1-e93642789667)

![raspberry_camera](https://github.com/user-attachments/assets/c243f4a5-1a0e-48db-b00f-42197850fbcb)

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Supported Crops & Diseases](#supported-crops--diseases)
- [How It Works](#how-it-works)
- [Testing & Quality](#testing--quality)
- [Raspberry Pi Setup](#raspberry-pi-setup)
- [Troubleshooting](#troubleshooting)
- [Educational Purpose](#educational-purpose)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features

- **Automated disease detection** — quantized MobileNet identifies 16 disease classes with 90%+ confidence on well-captured images, returning a ranked top-3 with per-class probabilities
- **Treatment recommendations** — symptoms, treatment, and prevention guidance for all 44 diseases in a schema-validated knowledge base
- **Zero-install browser demo** — live camera, drag-and-drop upload, clipboard paste, or one-click sample images
- **Disease Library** — every supported condition as a photo card, filterable by crop, with a detail modal of full care guidance
- **Recent analyses** — a local, restorable history of past diagnoses persisted in `localStorage`, never transmitted
- **Runs anywhere** — Raspberry Pi + camera module, any laptop with a webcam, or any modern browser
- **Private by design** — inference is 100% on-device; no server, no uploads, no third-party requests at runtime
- **Fast inference** — ~30 ms on a laptop, ~145 ms on Raspberry Pi 4, and multi-threaded WebAssembly in-browser on repeat visits
- **Out-of-distribution guard** — predictive-entropy and vegetation-coverage heuristics downgrade non-leaf images to *Uncertain* instead of returning a confident wrong answer
- **Installable PWA** — fully offline-capable after first visit (model, runtime, fonts, and photos all cached); installable to a phone home screen
- **Accessible by construction** — focus-trapped modals, live-region result announcements, keyboard-navigable throughout
- **Tested in CI** — golden prediction tests over real sample images, data-integrity tests across both runtimes, and a headless-browser end-to-end smoke test on every push

---

## Technology Stack

#### Browser Demo
| Technology | Role |
|---|---|
| ONNX Runtime Web (WASM + SIMD) | In-browser inference, multi-threaded when cross-origin isolated |
| Vanilla JS (no framework) | ~600 LOC of application code, zero runtime dependencies |
| Canvas 2D API | Frame capture, resize to tensor, vegetation analysis |
| Service Worker | Offline caching plus COOP/COEP header injection for WASM threads |
| Web App Manifest | Installable PWA with maskable icons |
| GitHub Pages | Zero-cost static hosting, no backend to operate |

#### Desktop App
| Technology | Role |
|---|---|
| Python 3.7+ | Application runtime |
| TensorFlow Lite / LiteRT / tflite-runtime | Quantized inference, resolved at import with graceful fallback |
| PyQt5 | Desktop GUI with live camera preview |
| OpenCV | Webcam capture, colour conversion, bilinear resize |
| Picamera2 | Raspberry Pi CSI camera module support |

#### Model & Data
| Asset | Details |
|---|---|
| MobileNet (uint8-quantized) | 11 MB, `[1,300,300,3]` input, 16-class output |
| [Kaggle AgriPredict](https://www.kaggle.com/models/agripredict/disease-classification) | Pretrained disease-classification model |
| `treatments.json` | Structured knowledge base, 44 diseases, CI-validated schema |
| tf2onnx | TFLite → ONNX conversion, verified to produce identical predictions |
| PlantVillage | CC0 leaf imagery for samples, library cards, and golden tests |

#### Quality & Tooling
| Technology | Role |
|---|---|
| pytest | Golden prediction and data-integrity suites |
| Playwright | Headless-Chromium end-to-end smoke test |
| GitHub Actions | Two-job CI matrix across the Python and browser runtimes |

---

## Getting Started

### Prerequisites

- **Browser demo**: any modern browser — nothing else
- **Desktop app**: Python 3.7+, and optionally a webcam
- **Raspberry Pi**: Pi 4 Model B with a camera module (see [Raspberry Pi Setup](#raspberry-pi-setup))

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

### Capture Tips

| Factor | Recommendation |
|---|---|
| Distance | 20–30 cm from the leaf |
| Lighting | Natural daylight or bright LED; avoid shadows |
| Framing | A single in-focus leaf filling most of the frame |
| Angle | Perpendicular to the leaf surface |
| Background | Plain and contrasting |

---

## Development Workflow

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

## Project Structure

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

## Core Components

The desktop and browser stacks are deliberate mirrors of each other — each layer has a counterpart on the other side, so a change in inference behaviour has exactly one obvious twin to update.

| Component | Role | Counterpart |
|---|---|---|
| `main.py` | Composition root: initializes camera, model, and database, then launches the GUI | — |
| `camera_module.py` | Unified capture API abstracting Picamera2 (CSI) and OpenCV (USB/built-in) behind one interface | `getUserMedia` in `app.js` |
| `ml_module.py` | Interpreter resolution, preprocessing, inference, dequantization, top-N ranking | `docs/js/inference.js` |
| `disease_database.py` | Loads and queries `treatments.json`; formats care guidance | `LeafModel.getTreatment()` |
| `gui_module.py` | PyQt5 window: live preview, capture, results panel, file loading | `docs/js/app.js` |
| `docs/js/inference.js` | ONNX Runtime Web session, tensor construction, entropy + vegetation scoring | `ml_module.py` |
| `docs/js/app.js` | Input modes, result rendering, history, disease library, modal | `gui_module.py` |
| `docs/sw.js` | Offline cache tiers and cross-origin isolation headers | — |

**Runtime resolution.** `ml_module.py` probes for `ai_edge_litert`, then `tensorflow`, then `tflite_runtime`, taking whichever is present. This is what lets the same file run under a 500 MB TensorFlow install on a laptop and a few-megabyte LiteRT wheel on a Raspberry Pi without conditional imports at the call site.

---

## Supported Crops & Diseases

![plant_dataset](https://github.com/user-attachments/assets/d8b5b407-6ce5-4f6e-a7d3-cbe942a3c1dd)

| Crop | Classes | Conditions | Pathogen types |
|---|---|---|---|
| Tomato | 8 | Healthy · Bacterial Spot · Septoria Leaf Spot · **Late Blight** · Leaf Mold · Spider Mites · **Yellow Leaf Curl Virus** | Bacterial, fungal, viral, arachnid |
| Corn (maize) | 4 | Healthy · Common Rust · Gray Leaf Spot · **Lethal Necrosis** | Fungal, viral |
| Soybean | 3 | Healthy · Frogeye Leaf Spot · Downy Mildew | Fungal, oomycete |
| Cabbage | 2 | Healthy · Black Rot | Bacterial |

**Bold** = critical severity. **Total**: 16 classes across 4 crops, spanning bacterial, fungal, oomycete, viral, and arthropod damage. The treatment knowledge base covers **44 diseases** — the extras (Apple, Grape, Potato, Strawberry, and more) are already schema-complete and ready for a future model trained on the full PlantVillage corpus.

Each knowledge-base entry is a structured record validated in CI:

```json
{
  "common_name": "Corn Common Rust",
  "plant": "Corn (maize)",
  "disease": "Common Rust",
  "severity": "medium",
  "description": "Fungal disease creating rust-colored pustules on leaves.",
  "symptoms":   ["Small reddish-brown pustules on both leaf surfaces", "..."],
  "treatments": ["Apply fungicides if disease is severe", "..."],
  "prevention": ["Use resistant hybrids", "..."]
}
```

---

## How It Works

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

### Model Specification

| Property | Value |
|---|---|
| Architecture | MobileNet (depthwise-separable CNN, edge-optimized) |
| Input tensor | `[1, 300, 300, 3]`, `uint8`, NHWC, raw RGB |
| Output tensor | `[1, 16]`, `uint8` affine-quantized scores |
| Quantization | Full-integer (weights **and** activations), ~4× smaller than float32 |
| Artifact size | 11 MB (`.tflite` and `.onnx` alike) |
| Preprocessing | Bilinear resize only — **no mean/std normalization** |
| Confidence floor | 0.30, applied after dequantization |

Because the network is fully integer-quantized, activations stay in `uint8` end to end and outputs are rescaled to probabilities with a single `v / 255` division. This is what makes ~145 ms inference on a Raspberry Pi 4 CPU — with no accelerator — achievable.

### Cross-Runtime Parity

Shipping one network through two runtimes only works if preprocessing matches bit-for-bit. Two decisions enforce that:

- **Identical resize semantics.** `cv2.resize` (desktop) and `ctx.drawImage` onto a 300×300 canvas (browser) both perform a plain bilinear resize with no aspect-ratio padding, so the tensors agree.
- **No normalization anywhere.** The quantized graph bakes the scale/zero-point into the model itself, so neither runtime applies ImageNet statistics. `ml_module.py` keeps a float32 normalization branch purely as a fallback for swapping in a non-quantized model later.

Parity is not assumed — `tests/test_treatments.py` asserts that the desktop and web label/knowledge-base files are byte-identical, so a change to one that isn't mirrored fails CI.

### Out-of-Distribution Detection

A 16-class softmax is *always* confident about something: photograph a rose, a keyboard, or a wall and the model still returns a tomato disease at 90%+. LeafMedic guards against this with two cheap signals computed during the same canvas pass as preprocessing (zero extra inference cost):

**1. Normalized predictive entropy** — how spread out the distribution is, scaled to `[0, 1]`:

```
H(p) = -Σ pᵢ · log(pᵢ) / log(N)      N = 16 classes
```

`H → 0` means one dominant class; `H → 1` means the model is guessing uniformly. Anything above **0.75** is flagged.

**2. Vegetation coverage** — the fraction of pixels whose channel relationships look like plant tissue (`G ≥ B`, `G ≥ R − 20`, `G > 40`). The tolerance is deliberately loose so that chlorotic yellow and necrotic brown leaves still register as vegetation, while sky, skin, pavement, and UI screenshots do not. Below **12 %** coverage the image is reported as "not a leaf."

Either trigger — or a top-1 confidence under the 0.30 floor — downgrades the verdict to **Uncertain** and swaps in guidance explaining what to reshoot, rather than presenting a confident wrong diagnosis.

### Browser Runtime Engineering

The demo is a static site on GitHub Pages, which imposes real constraints that shaped the implementation:

- **Multi-threaded WASM without server headers.** SIMD-threaded ONNX Runtime requires `crossOriginIsolated`, which requires COOP/COEP response headers — and GitHub Pages cannot send custom headers. The service worker resolves this by intercepting same-origin responses and re-issuing them with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. First visit runs single-threaded; once the worker is in control, inference scales to `min(4, hardwareConcurrency)` threads. Every asset is same-origin, so `require-corp` breaks nothing.
- **Streamed download with real progress.** The 11 MB model is fetched through a `ReadableStream` reader that tracks `Content-Length` to drive an accurate progress bar, instead of a spinner that lies until the whole buffer lands.
- **Tiered caching.** The service worker is network-first for the app shell (so updates reach returning visitors immediately) and cache-first for immutable heavy assets — model, WASM binaries, fonts, and photos.
- **Zero third-party runtime dependencies.** ONNX Runtime Web and both typefaces are self-hosted; the page makes no CDN, analytics, or telemetry requests, which is what makes the privacy claim verifiable rather than promotional.

### Performance

| Platform | Runtime | Inference (300×300) |
|---|---|---|
| Laptop (x86-64) | TFLite + XNNPACK | ~30 ms |
| Raspberry Pi 4B | TFLite (LiteRT) | ~145 ms |
| Browser, first visit | ONNX Runtime Web, 1 thread | ~120–200 ms |
| Browser, repeat visit | ONNX Runtime Web, up to 4 threads | ~60–90 ms |

Model load is a one-time ~11 MB download, cached indefinitely by the service worker; a warm-up inference on a zero tensor is issued at startup so the first real diagnosis never pays allocation cost.

---

## Testing & Quality

Both runtimes are exercised on every push and pull request by [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

| Suite | What it protects |
|---|---|
| `tests/test_ml.py` | **Golden predictions** — every image in `test_images/` is classified, and each class must be top-1 for a majority of its samples. Catches silent model, preprocessing, or dequantization regressions. |
| `tests/test_ml.py` | Confidence outputs are valid probabilities in `[0, 1]` and map to known labels |
| `tests/test_treatments.py` | Every model label has a knowledge-base entry; no diagnosis can render without care guidance |
| `tests/test_treatments.py` | Desktop and web copies of `labels`/`treatments` stay identical; severity values are from a known set; disease records carry all required fields |
| `tests/web_smoke.mjs` | **End-to-end browser run** — serves `docs/`, drives headless Chromium, executes a real diagnosis, opens the library modal, and fails on any console error |

The golden tests use majority-vote rather than per-image assertions: individual field photographs legitimately vary, but a genuine regression moves the whole class at once. The Python suite skips itself gracefully when no TFLite runtime is installed, so contributors without the ML stack can still run the data-integrity tests.

---

## Raspberry Pi Setup

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

## Troubleshooting

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

## Educational Purpose

This project demonstrates:

- **Edge AI deployment** — one quantized network served through three runtimes (Pi, desktop, browser/WASM) with verified cross-runtime parity
- **Model quantization trade-offs** — full-integer `uint8` inference, dequantization arithmetic, and the size/latency wins that make CPU-only inference viable
- **Uncertainty quantification** — predictive entropy and domain heuristics as a practical answer to the closed-set softmax problem
- **Browser runtime engineering** — WebAssembly threading, cross-origin isolation, streamed model loading, and service-worker cache tiering
- **Computer vision** — capture, colour-space conversion, and resize pipelines in both OpenCV and the Canvas API
- **GUI development** — PyQt5 alongside a hand-built, dependency-free responsive web UI
- **Modular architecture** — mirrored Python and JavaScript modules with explicit counterparts
- **Test strategy for ML systems** — golden-output regression tests and data-contract validation, where conventional unit tests do not apply

> **Disclaimer:** This is a learning project, not a professional diagnostic tool. The model covers 4 crops / 16 classes — unsupported plants yield meaningless predictions, and visually similar diseases can be confused. For real crop management decisions, consult agricultural extension services or plant pathologists.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

Ideas that would be especially welcome: PlantVillage-trained models with more crops (38+ classes), better treatment recommendations, UI improvements in either app, and bug fixes.

---

## License

**MIT** — see [LICENSE](LICENSE). Built as an educational project; not a professional diagnostic tool.

- **Model**: Kaggle AgriPredict Disease Classification (16 classes)
- **Dataset (future expansion)**: PlantVillage — CC0 Public Domain
- **Treatment data**: compiled from public agricultural resources
- **Dependencies**: TensorFlow (Apache 2.0) · ONNX Runtime (MIT) · PyQt5 (GPL v3) · OpenCV (Apache 2.0) · Nunito & Outfit fonts (OFL 1.1)

---

## Acknowledgments

- **AgriPredict** — Disease Classification TFLite model on Kaggle
- **PlantVillage** — 54,000+ image dataset for future model training
- **TensorFlow** & **ONNX Runtime** — inference runtimes
- **Raspberry Pi Foundation** — hardware platform and Picamera2
- **SunFounder** — the electronics kit where this project began

---

## Contact

If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).

*Educational Project — Learn, Experiment, Innovate*
