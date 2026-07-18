# LeafMedic: Plant Disease Detection — On-Device AI

[![CI](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml) ![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20it%20now-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blueviolet) ![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-brightgreen) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red) ![Edge AI](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![License](https://img.shields.io/badge/License-MIT-yellowgreen)

**LeafMedic** identifies plant diseases from a photo of a leaf and explains the symptoms, treatment, and prevention for what it finds. A fully integer-quantized MobileNet CNN classifies **16 conditions across 4 crops** entirely **on-device** — no server, no uploads, no photo ever leaves your machine.

The same network is deployed through three runtimes from a single source of truth: a **browser app** (ONNX Runtime Web on WebAssembly), a **desktop app** (Python + PyQt5 with any webcam), and its original target, a **Raspberry Pi 4** with a camera module. Every diagnosis is paired with treatment and prevention guidance from a structured knowledge base covering 44 diseases.

It is built as a study in **edge AI deployment constraints**: an 11 MB quantized graph converted across two inference runtimes with prediction parity asserted in CI, sub-150 ms CPU inference on a Raspberry Pi, WebGPU acceleration with a WebAssembly fallback, multi-threaded WASM unlocked on a static host that cannot set HTTP headers, and layered guards so the classifier declines to guess instead of confidently misdiagnosing an unsupported plant.

Every performance and accuracy claim below is reproducible with a command in this repository — `benchmark.py` for latency, `training/evaluate.py` for per-class accuracy, and `pytest` for cross-runtime parity.

---

## Live Demo

**[mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)**

> **Note:** The model (~11 MB) downloads once on first visit and is cached afterwards — the page even works offline and can be installed to your phone's home screen.

<img width="2772" height="1504" alt="Demo" src="https://github.com/user-attachments/assets/f37e3914-394d-4425-a2e0-100fdd1cd5cf" />

<img width="1470" height="792" alt="Screenshot 2026-07-17 at 7 56 57 PM" src="https://github.com/user-attachments/assets/1904791b-f9ef-4298-bdc2-3ee97d777f37" />

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
- [Training Your Own Model](#training-your-own-model)
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
- **Capture-quality feedback** — blur (Laplacian variance) and exposure checks tell you to retake a photo *before* a bad diagnosis, in both the browser and the desktop app
- **Explainable predictions** — a "Why this diagnosis?" heatmap uses occlusion sensitivity to show which leaf regions the model actually relied on
- **WebGPU acceleration** — 4.5x faster in-browser inference where a GPU is available, falling back to WebAssembly everywhere else
- **Bilingual** — full English and Spanish interface *and* care guidance for all 44 diseases, switchable at runtime
- **Installable PWA** — fully offline-capable after first visit (model, runtime, fonts, and photos all cached); installable to a phone home screen
- **Accessible by construction** — focus-trapped modals, live-region result announcements, keyboard-navigable throughout
- **Tested in CI** — 146 tests: golden predictions, exact TFLite/ONNX output parity, out-of-distribution and capture-quality guards, data integrity, plus a headless-browser end-to-end run on every push
- **Retrainable** — a complete training pipeline (`training/`) for fine-tuning on PlantVillage's 38 classes, with int8 quantization, ONNX export, and parity verification built in

---

## Technology Stack

#### Browser Demo
| Technology | Role |
|---|---|
| ONNX Runtime Web (WebGPU + WASM) | In-browser inference; WebGPU when a GPU is available, multi-threaded WASM otherwise |
| Vanilla JS (no framework) | ~1,100 LOC of application code, zero runtime dependencies |
| Canvas 2D API | Frame capture, resize to tensor, quality and vegetation analysis |
| Service Worker | Offline caching plus COOP/COEP header injection for WASM threads |
| Web App Manifest | Installable PWA with maskable icons |
| GitHub Pages | Zero-cost static hosting, no backend to operate |
| Hand-rolled i18n | English/Spanish UI and knowledge base, no framework |

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
| tf2onnx | TFLite → ONNX conversion; identical outputs asserted by `tests/test_parity.py` |
| PlantVillage | CC0 leaf imagery for samples, library cards, and golden tests |

#### Quality & Tooling
| Technology | Role |
|---|---|
| pytest | Golden predictions, cross-runtime parity, quality guards, data integrity (146 tests) |
| Playwright | Headless-Chromium end-to-end smoke test |
| ruff + mypy | Lint and type checking, enforced in CI |
| GitHub Actions | Three-job CI: lint/types, Python tests, browser smoke test |

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
| `pytest tests/` | Runs the full Python suite (golden, parity, quality, data integrity) |
| `node tests/web_smoke.mjs` | End-to-end browser demo smoke test (needs `playwright`) |
| `python3 benchmark.py --onnx` | Measures inference latency on your machine |
| `python3 image_quality.py <img>` | Prints the quality/OOD metrics for an image |
| `python3 tests/make_negatives.py` | Regenerates the synthetic non-leaf test images |
| `python3 tools/build_translations.py` | Rebuilds the Spanish knowledge base |
| `ruff check . && mypy *.py` | Lint and type check |

---

## Project Structure

```
LeafMedic/
├── main.py                      # Desktop app entry point (run this!)
├── camera_module.py             # Camera control: Picamera2 with OpenCV webcam fallback
├── ml_module.py                 # ML inference with TensorFlow Lite
├── image_quality.py             # Blur, exposure, vegetation and entropy guards
├── gui_module.py                # PyQt5 graphical interface
├── disease_database.py          # Disease information management
├── benchmark.py                 # Reproducible latency benchmark
├── download_model.py            # Instructions/helper for obtaining the model
├── pyproject.toml               # Packaging, ruff, mypy and pytest configuration
├── requirements.txt             # Python dependencies (desktop)
├── requirements-pi.txt          # Lighter dependencies for Raspberry Pi
├── tests/                       # Pytest suites + browser smoke test (run in CI)
├── tools/build_translations.py  # Generates the translated knowledge bases
├── training/                    # Model training pipeline (see training/README.md)
│   ├── download_dataset.py      # Fetches PlantVillage (CC0)
│   ├── train.py                 # Fine-tune, quantize to int8, export ONNX
│   └── evaluate.py              # Per-class metrics + confusion matrix
├── .github/workflows/ci.yml    # CI: lint/types, Python tests, browser smoke test
├── models/
│   ├── plant_disease_model.tflite  # AgriPredict model (11 MB, 16 classes)
│   └── labels.txt               # 16 class labels
├── data/
│   └── treatments.json          # Disease treatment database (44 diseases)
├── test_images/                 # Real sample images, plus _negatives/ for OOD tests
└── docs/                        # Browser demo (GitHub Pages)
    ├── index.html               # Single-page app
    ├── js/inference.js          # Backend selection, model loading, prediction, occlusion maps
    ├── js/quality.js            # Quality/OOD heuristics (mirrors image_quality.py)
    ├── js/i18n.js               # UI string tables and language switching
    ├── js/app.js                # Camera, upload, results, history, disease library
    ├── model/leafmedic.onnx     # Same network, converted for the browser
    ├── data/treatments.es.json  # Spanish knowledge base (generated)
    ├── sw.js                    # Service worker (offline + COOP/COEP for threads)
    ├── fonts/                   # Self-hosted Nunito & Outfit (offline-safe)
    ├── img/                     # Hero illustration + disease-cause card images
    └── vendor/                  # ONNX Runtime Web, WASM + WebGPU builds (no CDN)
```

---

## Core Components

The desktop and browser stacks are deliberate mirrors of each other — each layer has a counterpart on the other side, so a change in inference behaviour has exactly one obvious twin to update.

| Component | Role | Counterpart |
|---|---|---|
| `main.py` | Composition root: initializes camera, model, and database, then launches the GUI | — |
| `camera_module.py` | Unified capture API abstracting Picamera2 (CSI) and OpenCV (USB/built-in) behind one interface | `getUserMedia` in `app.js` |
| `ml_module.py` | Interpreter resolution, preprocessing, inference, dequantization, top-N ranking | `docs/js/inference.js` |
| `image_quality.py` | Vegetation coverage, entropy, blur and exposure guards | `docs/js/quality.js` |
| `disease_database.py` | Loads and queries `treatments.json`; formats care guidance | `LeafModel.getTreatment()` |
| `gui_module.py` | PyQt5 window: live preview, capture, results panel, file loading | `docs/js/app.js` |
| `docs/js/inference.js` | Backend selection, ORT session, tensor construction, occlusion maps | `ml_module.py` |
| `docs/js/quality.js` | Single-pass pixel scan: vegetation, luma, clipping, Laplacian variance | `image_quality.py` |
| `docs/js/i18n.js` | UI string tables, language detection and persistence | — |
| `docs/js/app.js` | Input modes, result rendering, history, disease library, modal | `gui_module.py` |
| `docs/sw.js` | Offline cache tiers and cross-origin isolation headers | — |

**Runtime resolution.** `ml_module.py` probes for `ai_edge_litert`, then `tensorflow`, then `tflite_runtime`, taking whichever is present. This is what lets the same file run under a 500 MB TensorFlow install on a laptop and a few-megabyte LiteRT wheel on a Raspberry Pi without conditional imports at the call site.

---

<img width="1454" height="789" alt="Screenshot 2026-07-17 at 7 53 17 PM" src="https://github.com/user-attachments/assets/b7e7fd1c-bd15-414d-b8cf-3962d081ef66" />

<img width="1470" height="801" alt="Screenshot 2026-07-17 at 7 54 01 PM" src="https://github.com/user-attachments/assets/c1562c65-35bf-4e26-a50c-168ba0d7b0f7" />


---

## Supported Crops & Diseases


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

### Trusting a Prediction

A 16-class softmax is *always* confident about something: photograph a rose, a keyboard, or a wall and the model still returns a tomato disease at 90%+. LeafMedic answers three separate questions before showing a diagnosis, all computed in the same canvas/OpenCV pass as preprocessing, at zero extra inference cost.

**1. Is this a leaf at all? — vegetation coverage.** The fraction of pixels whose colour looks like plant tissue: hue within a yellow-to-green window, with a minimum saturation and value.

```
vegetation = mean( 15 <= H <= 100  and  S >= 45  and  V >= 40 )     H,S,V on OpenCV scales
```

Below **12 %** coverage the image is reported as "not a leaf". The saturation floor is what does the real work — an earlier version of this heuristic compared raw channels (`G >= B and G >= R - 20`), which every neutral grey satisfies, so concrete scored 100 % vegetation and a UI screenshot 79 %. Both now score 0.000, while diseased chlorotic-yellow and necrotic-brown tissue still registers as vegetation. `tests/test_quality.py` asserts both directions.

**2. Is the model actually sure? — normalized predictive entropy.**

```
H(p) = -Σ pᵢ · log(pᵢ) / log(N)      N = 16 classes
```

`H → 0` means one dominant class; `H → 1` means the model is guessing uniformly. Anything above **0.75**, or a top-1 confidence under the **0.30** floor, is flagged.

**3. Is the photo usable? — blur and exposure.** Variance of the Laplacian is the standard focus measure: sharp edges produce a wide second-derivative distribution, a defocused image a narrow one. Below **10** the photo is called out of focus. Mean luma outside **45–215**, or more than 35 % of pixels clipped to pure black or white, flags exposure.

This third check is different in kind from the other two, and is surfaced separately in the UI: "not a leaf" means the answer is meaningless, while "too blurry" is *actionable* — it tells the user exactly what to do differently, before they act on a bad diagnosis.

Any of these tripping downgrades the verdict to **Uncertain** and swaps in guidance explaining what to reshoot. The thresholds live in exactly two places — `image_quality.py` and `docs/js/quality.js` — and a test parses the JavaScript to assert the numbers match, because a drift would make the desktop app and the browser disagree about which photos to trust.

**A known gap, stated plainly:** uniform RGB noise defeats all three heuristics. It contains genuinely leaf-coloured pixels and is extremely sharp, so it passes. A Laplacian *ceiling* cannot separate it from a detailed real photograph (real leaves here reach ~19,900 variance against noise's ~48,800); rejecting it properly needs a spatial-coherence signal. `test_uniform_noise_is_a_known_gap` documents this rather than hiding it.

### Explaining a Prediction

Confidence bars say *how sure* the model is, not *why*. The **"Why this diagnosis?"** button answers the second question with **occlusion sensitivity**: slide an opaque patch across the image, re-run inference at each position, and measure how far the predicted class's confidence falls. Regions whose removal hurts the prediction most are the ones the network relied on.

The shipped graph is fully integer-quantized and exposes neither gradients nor intermediate activations, so Grad-CAM is not an option. Occlusion needs nothing but repeated forward passes, which makes it the practical choice here — 64 extra inferences on an 8×8 grid, with patches overlapping at 1.5 cells so features straddling a cell boundary are not missed. On WebGPU that is roughly a second.

### Browser Runtime Engineering

The demo is a static site on GitHub Pages, which imposes real constraints that shaped the implementation:

- **WebGPU when it pays for itself.** ONNX Runtime ships separate bundles per backend, and the WebGPU-capable one needs a 5.6 MB gzipped WASM binary against 3.2 MB for the WASM-only build. Rather than guess, both were benchmarked on this model: WebGPU runs at a **14.3 ms median against 63.7 ms** for multi-threaded WASM — 4.5x, with a bit-identical output checksum. Worth 2.4 MB, so the demo probes for a real GPU adapter (`navigator.gpu` can exist while `requestAdapter()` returns null) and injects the matching bundle at load time. Browsers without WebGPU never download the larger binary, and a browser that advertises an adapter but fails to create a device falls back to WASM instead of erroring.
- **Multi-threaded WASM without server headers.** SIMD-threaded ONNX Runtime requires `crossOriginIsolated`, which requires COOP/COEP response headers — and GitHub Pages cannot send custom headers. The service worker resolves this by intercepting same-origin responses and re-issuing them with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. First visit runs single-threaded; once the worker is in control, inference scales to `min(4, hardwareConcurrency)` threads. Every asset is same-origin, so `require-corp` breaks nothing.
- **Streamed download with real progress.** The 11 MB model is fetched through a `ReadableStream` reader that tracks `Content-Length` to drive an accurate progress bar, instead of a spinner that lies until the whole buffer lands.
- **Tiered caching.** The service worker is network-first for the app shell (so updates reach returning visitors immediately) and cache-first for immutable heavy assets — model, WASM binaries, fonts, and photos.
- **Zero third-party runtime dependencies.** ONNX Runtime Web and both typefaces are self-hosted; the page makes no CDN, analytics, or telemetry requests, which is what makes the privacy claim verifiable rather than promotional.
- **Warm-up sized to the backend.** WebGPU compiles shaders lazily, so a single warm-up pass leaves ~300 ms of that cost for the user's first real photo. Three passes on a zero tensor at load time move it off the critical path.

### Performance

Measured on an Apple Silicon laptop with `python3 benchmark.py --onnx` and the browser's `?bench` mode. Medians over 25–40 runs after warm-up:

| Platform | Runtime | Inference (300×300) |
|---|---|---|
| Laptop, desktop app | TFLite (LiteRT) | **29 ms** |
| Laptop, desktop app | ONNX Runtime (Python) | **24 ms** |
| Laptop, browser | ONNX Runtime Web, WebGPU | **14 ms** |
| Laptop, browser | ONNX Runtime Web, WASM | **64 ms** |
| Raspberry Pi 4B | TFLite (LiteRT) | ~145 ms |

Reproduce them yourself rather than trusting the table:

```bash
python3 benchmark.py --onnx --runs 100        # desktop runtimes
# browser: open the demo with ?bench, or run leafmedicBenchmark(50) in the console
```

Model load is a one-time ~11 MB download, cached indefinitely by the service worker.

---

## Testing & Quality

Every push and pull request runs three jobs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml): lint and type checking, the Python suite (146 tests), and a headless browser run.

| Suite | What it protects |
|---|---|
| `tests/test_ml.py` | **Golden predictions** — each sample class must be top-1 for a majority of its images. Catches silent model, preprocessing, or dequantization regressions. |
| `tests/test_ml.py` | `analyze()` pairs predictions with quality signals, and flags a non-leaf image as untrustworthy |
| `tests/test_parity.py` | **Cross-runtime parity** — TFLite and ONNX must produce *byte-identical* outputs on every sample image and on random tensors, plus matching I/O signatures |
| `tests/test_quality.py` | **The guards themselves** — real leaves pass, synthetic sky/skin/concrete/screenshots are rejected, greys are not vegetation, diseased yellow and brown tissue still is, entropy is bounded, thresholds match between Python and JavaScript |
| `tests/test_treatments.py` | Every model label has care guidance; desktop and web copies stay identical; severities are from a known set |
| `tests/test_quality.py` | Every translated knowledge base covers the full label set with matching severities and list lengths |
| `tests/web_smoke.mjs` | **End-to-end browser run** — real diagnosis, occlusion heatmap, OOD rejection of a non-leaf image, Spanish/English switch, library modal, and zero console errors |

The golden tests use majority-vote rather than per-image assertions: individual field photographs legitimately vary, but a genuine regression moves the whole class at once. The Python suite skips gracefully when no TFLite runtime is installed, so contributors without the ML stack can still run the data-integrity tests.

### What the model gets wrong

Broadening the test corpus from 3 classes to 11 surfaced real weaknesses in the pretrained AgriPredict model, and the suite records them rather than quietly omitting them. Measured on PlantVillage imagery, three images per class:

| Class | Top-1 correct | Predicted instead |
|---|---|---|
| Tomato — Septoria, Late blight, Spider mites, Yellow leaf curl | 3/3 each | — |
| Corn — Common rust | 3/3 | — |
| Tomato — healthy | 1/3 | Spider mites, Septoria leaf spot |
| Corn — Gray leaf spot | 0/3 | Common rust |
| Corn — healthy | 0/3 | Gray leaf spot, Tomato late blight |
| Soybean — healthy | 0/3 | Tomato spider mites, at 98 % confidence |

The four failing classes are marked `xfail` in `tests/test_ml.py` rather than deleted, so the gap stays visible and a retrained model that fixes one reports an unexpected pass instead of going unnoticed. **Healthy foliage is the weak spot** — which is unfortunate, because a healthy leaf is exactly what a worried grower photographs first.

Some of this is domain shift: AgriPredict trained on its own corpus, not PlantVillage. That is precisely why [`training/`](training/README.md) exists — a model trained on the data it is evaluated against is the fix, and the out-of-distribution guard limits the damage in the meantime.

Reproduce the per-class breakdown with a confusion matrix:

```bash
python3 training/download_dataset.py --per-class 50
python3 training/evaluate.py --model models/plant_disease_model.tflite --labels models/labels.txt
```

---

## Training Your Own Model

The shipped model is pretrained; [`training/`](training/README.md) is the pipeline for replacing it. It fine-tunes an ImageNet backbone on PlantVillage's 38 classes, quantizes to full-integer uint8 with a real calibration set, exports both `.tflite` and `.onnx`, and verifies the two agree before you deploy them.

```bash
pip install -e '.[train]'
python3 training/download_dataset.py --per-class 200
python3 training/train.py --epochs 15 --fine-tune-epochs 8
python3 training/evaluate.py --model training/runs/<run>/model_int8.tflite
```

There is headroom waiting: `data/treatments.json` already covers **all 38 PlantVillage classes**, so the gap closes with no new care guidance to write. See [`training/README.md`](training/README.md) for the design decisions — why preprocessing lives inside the graph, why quantization calibration needs real images, and what to check when swapping a new model in.

**One caveat, measured rather than assumed:** the pipeline runs end to end, but the TFLite→ONNX conversion is *not* bit-exact for models it produces, unlike the shipped one. MobileNetV2 lands within 6 quantization steps of 255, MobileNetV3-Small within 18. That is small enough not to change a confident diagnosis and large enough to matter before promising the browser and desktop apps agree exactly. [`training/README.md`](training/README.md#open-issue-onnx-conversion-is-not-bit-exact-for-new-models) documents the numbers and what to check before deploying.

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
- Browsers with WebGPU fetch a larger runtime (5.6 MB gzipped vs 3.2 MB) in exchange for ~4.5x faster inference

**Checking which backend is in use**
- Open the console: the page logs `[LeafMedic] inference backend: webgpu` or `wasm`
- Or load the demo with `?bench` for a live latency readout

**The interface is in the wrong language**
- Use the language selector in the top-right; the choice is remembered
- Without a stored choice the page follows your browser's language, falling back to English

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
- **Explainability without gradients** — occlusion sensitivity on a black-box quantized graph where Grad-CAM cannot reach
- **Measuring instead of assuming** — a WebGPU backend adopted because a benchmark contradicted the expectation that it would not help on an int8 model
- **Browser runtime engineering** — WebAssembly threading, cross-origin isolation, streamed model loading, and service-worker cache tiering
- **Computer vision** — capture, colour-space conversion, and resize pipelines in both OpenCV and the Canvas API
- **GUI development** — PyQt5 alongside a hand-built, dependency-free responsive web UI
- **Modular architecture** — mirrored Python and JavaScript modules with explicit counterparts
- **Test strategy for ML systems** — golden-output regression tests, cross-runtime parity assertions, and data-contract validation, where conventional unit tests do not apply
- **Reporting negative results** — documenting the classes the model fails and the inputs the guards miss, rather than reporting only what works

> **Disclaimer:** This is a learning project, not a professional diagnostic tool. The model covers 4 crops / 16 classes — unsupported plants yield meaningless predictions, and visually similar diseases can be confused. For real crop management decisions, consult agricultural extension services or plant pathologists.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

Ideas that would be especially welcome: running the [`training/`](training/README.md) pipeline to produce a 38-class model that fixes the healthy-leaf weakness, additional languages (add a table to `docs/js/i18n.js` and a `treatments.<lang>.json`), a spatial-coherence signal that catches the noise gap in the OOD guard, better treatment recommendations, and UI improvements in either app.

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
