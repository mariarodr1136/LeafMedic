# LeafMedic — On-Device Plant Disease Detection

[![CI](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml/badge.svg)](https://github.com/mariarodr1136/LeafMedic/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blueviolet) ![Edge AI](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![License](https://img.shields.io/badge/License-MIT-yellowgreen)

Snap a photo of a leaf and get a diagnosis — symptoms, treatment, and prevention — for **16 conditions across 4 crops**. A fully integer-quantized MobileNet runs **entirely on-device**: no server, no uploads, no photo ever leaves your machine.

One network, three runtimes, single source of truth: a **browser app** (ONNX Runtime Web — WebGPU with WASM fallback), a **desktop app** (Python + PyQt5), and a **Raspberry Pi 4** with a camera module. Cross-runtime prediction parity is asserted in CI, and every performance claim below is reproducible with a command in this repo.

Live Demo: [mariarodr1136.github.io/LeafMedic](https://mariarodr1136.github.io/LeafMedic/)

Works in any modern browser — installable as an offline-capable PWA after the one-time 11 MB model download.

<img width="2772" height="1504" alt="LeafMedic browser demo" src="https://github.com/user-attachments/assets/f37e3914-394d-4425-a2e0-100fdd1cd5cf" />

<img width="1470" height="792" alt="Diagnosis with treatment guidance" src="https://github.com/user-attachments/assets/1904791b-f9ef-4298-bdc2-3ee97d777f37" />

![Raspberry Pi with camera module](https://github.com/user-attachments/assets/c243f4a5-1a0e-48db-b00f-42197850fbcb)

## Table of Contents

- [Highlights](#highlights)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Performance](#performance)
  - [Notable engineering decisions](#notable-engineering-decisions)
- [Supported Crops](#supported-crops)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License & Credits](#license--credits)
- [Contact](#contact)

## Highlights

- **Edge AI deployment** — an 11 MB uint8-quantized MobileNet converted from TFLite to ONNX, with byte-identical outputs verified by automated parity tests
- **Fast on CPU alone** — 14 ms in-browser (WebGPU), 64 ms (multi-threaded WASM), ~145 ms on a Raspberry Pi 4
- **Knows when not to answer** — predictive-entropy and vegetation heuristics downgrade non-leaf or ambiguous images to *Uncertain* instead of confidently misdiagnosing
- **Explainable predictions** — an occlusion-sensitivity heatmap shows which leaf regions drove the diagnosis (gradient-free, so it works on the quantized black-box graph)
- **Capture-quality feedback** — blur and exposure checks prompt a retake *before* a bad diagnosis
- **Multi-photo diagnosis** — upload 2–5 photos of the same plant and the predictions are averaged (shots that fail the leaf check are left out); any diagnosis can be saved as a PDF report from the browser's print dialog
- **Plant timelines** — tag a saved diagnosis with a plant name (e.g. "Backyard tomato #1") and repeat diagnoses of that plant line up in a chronological strip, so a recurring issue — or its recovery — is visible at a glance; all local, no account needed
- **Measured calibration** — `training/evaluate.py` reports Expected Calibration Error with a reliability diagram and dry-runs the trust-guard thresholds, so "knows when not to answer" is a number, not a slogan
- **Private by design** — zero third-party requests at runtime; runtime, fonts, and model are all self-hosted
- **Bilingual** — full English/Spanish UI and care guidance, switchable at runtime
- **167 tests in CI** — golden predictions, cross-runtime parity, quality guards, data-contract validation, JS unit tests for the browser-only logic, plus a headless-browser end-to-end smoke test on every push
- **Retrainable** — a complete pipeline (`training/`) for fine-tuning on PlantVillage, with int8 quantization, ONNX export, and parity verification built in

## Tech Stack

| Layer | Technologies |
|---|---|
| Browser | ONNX Runtime Web (WebGPU + WASM), vanilla JS (zero runtime dependencies), Canvas API, Service Worker PWA, GitHub Pages |
| Desktop / Pi | Python, TensorFlow Lite / LiteRT, PyQt5, OpenCV, Picamera2 |
| Model | MobileNet, full-integer uint8 quantization, 11 MB, 300×300 RGB input, TFLite → ONNX via tf2onnx |
| Quality | pytest (147 tests), `node:test` (20 tests), Playwright end-to-end, Lighthouse CI, ruff + mypy, five-job GitHub Actions CI |

## Quick Start

**Browser** — open the [live demo](https://mariarodr1136.github.io/LeafMedic/). That's it.

**Desktop:**

```bash
git clone https://github.com/mariarodr1136/LeafMedic.git
cd LeafMedic
pip3 install -r requirements.txt
python3 main.py
```

The app auto-detects a Raspberry Pi camera module or any webcam, and can also analyze image files — `test_images/` has real samples to try.

<details>
<summary><b>Raspberry Pi setup</b></summary>

Hardware: Pi 4 Model B (4 GB recommended) + camera module.

```bash
sudo raspi-config   # Interface Options → Camera → Enable
sudo apt update && sudo apt install python3-picamera2
pip3 install -r requirements-pi.txt --break-system-packages
python3 main.py
```

Full hardware list, camera connection steps, and troubleshooting: [CONTRIBUTING.md](CONTRIBUTING.md#raspberry-pi-setup).

</details>

## How It Works

```
   Pi camera / webcam / browser upload
                  │
                  ▼
   Preprocess → 300×300 RGB uint8
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   TFLite (desktop/Pi)   ONNX Runtime Web (browser)
        └─────────┬─────────┘
                  ▼
   Top-3 predictions + quality/OOD verdict
   Treatment guidance from knowledge base (44 diseases)
```

1. **Capture & preprocess** — bilinear resize to 300×300; the quantized graph bakes normalization into the model, so both runtimes feed raw uint8 pixels and their tensors match bit-for-bit.
2. **Inference** — full-integer MobileNet; activations stay uint8 end to end, which is what makes ~145 ms CPU inference on a Pi 4 achievable.
3. **Trust checks** — before showing a result, three guards run in the same pixel pass as preprocessing: vegetation coverage (is this a leaf?), normalized predictive entropy (is the model actually sure?), and blur/exposure (is the photo usable?). Any failure downgrades the verdict to *Uncertain* with reshoot guidance. Thresholds are mirrored between Python and JavaScript, and a test asserts they never drift apart.
4. **Explain** — the "Why this diagnosis?" heatmap slides an occlusion patch across the image and measures confidence drop per region — the practical choice for a quantized graph that exposes no gradients.

### Performance

Medians after warm-up; reproduce with `python3 benchmark.py --onnx` or the demo's `?bench` mode.

| Platform | Runtime | Inference |
|---|---|---|
| Laptop, browser | ONNX Runtime Web (WebGPU) | **14 ms** |
| Laptop, desktop | ONNX Runtime (Python) | **24 ms** |
| Laptop, desktop | TFLite (LiteRT) | **29 ms** |
| Laptop, browser | ONNX Runtime Web (WASM, 4 threads) | **64 ms** |
| Raspberry Pi 4B | TFLite (LiteRT) | ~145 ms |

### Notable engineering decisions

- **WebGPU adopted by measurement, not assumption** — benchmarked at 4.5× faster than multi-threaded WASM on this int8 model with a bit-identical output checksum, so the demo probes for a real GPU adapter and loads the matching runtime bundle.
- **Multi-threaded WASM on a host that can't set headers** — GitHub Pages can't send the COOP/COEP headers ONNX Runtime's threading requires, so the service worker injects them by re-issuing same-origin responses.
- **Honest evaluation** — the pretrained model's weak classes (healthy foliage, in particular) are documented and marked `xfail` in the test suite rather than hidden, and a known gap in the OOD guard is captured in a test that explains it.

## Supported Crops

| Crop | Conditions |
|---|---|
| Tomato | Healthy · Bacterial Spot · Septoria Leaf Spot · Late Blight · Leaf Mold · Spider Mites · Yellow Leaf Curl Virus |
| Corn (maize) | Healthy · Common Rust · Gray Leaf Spot · Lethal Necrosis |
| Soybean | Healthy · Frogeye Leaf Spot · Downy Mildew |
| Cabbage | Healthy · Black Rot |

The schema-validated knowledge base already covers **44 diseases** (all 38 PlantVillage classes), ready for an expanded model from the [training pipeline](training/README.md).

---



https://github.com/user-attachments/assets/f92c4ee8-e3b5-4afb-9fe6-94a5d07572e7



---

<img width="1470" height="799" alt="Screenshot 2026-07-18 at 2 02 42 PM" src="https://github.com/user-attachments/assets/bad17888-af53-449a-b00e-81648bd53271" />

---

## Project Structure

```
LeafMedic/
├── main.py               # Desktop app entry point
├── ml_module.py          # TFLite inference (mirrors docs/js/inference.js)
├── image_quality.py      # Blur/exposure/vegetation/entropy guards (mirrors docs/js/quality.js)
├── camera_module.py      # Picamera2 with OpenCV webcam fallback
├── gui_module.py         # PyQt5 interface
├── disease_database.py   # Treatment knowledge base (data/treatments.json)
├── training/             # Fine-tune, quantize, export, evaluate
├── tests/                # 147 pytest tests + 20 JS unit tests + Playwright browser smoke test
└── docs/                 # Browser demo (GitHub Pages, PWA)
```

The desktop and browser stacks are deliberate mirrors — every inference-path module has an exact counterpart on the other side, so a behavior change has one obvious twin to update.

## Testing

Every push runs five CI jobs: lint + type checking (ruff, mypy), the Python suite, JS unit tests for the browser-only logic, a headless-browser end-to-end run, and Lighthouse CI (accessibility and SEO gated at 0.9, performance and best-practices tracked as a warning).

- **Golden predictions** — each sample class must be top-1 by majority vote, catching silent model or preprocessing regressions
- **Cross-runtime parity** — TFLite and ONNX outputs must be byte-identical on every sample image and on random tensors
- **Guard behavior** — real leaves pass; synthetic sky, skin, concrete, and screenshots are rejected; diseased yellow/brown tissue still counts as vegetation
- **Data contracts** — every model label has care guidance, translations cover the full label set, desktop and web copies stay identical
- **End-to-end** — Playwright drives a real diagnosis, heatmap, OOD rejection, and language switch with zero console errors

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, architecture internals (trust-guard heuristics, cross-runtime parity, browser runtime engineering), full Raspberry Pi setup, and troubleshooting.

## License & Credits

**MIT** — see [LICENSE](LICENSE). Built as an educational project; not a professional diagnostic tool — for real crop decisions, consult agricultural extension services.

Model: [Kaggle AgriPredict](https://www.kaggle.com/models/agripredict/disease-classification) · Dataset: PlantVillage (CC0) · Runtimes: TensorFlow Lite, ONNX Runtime

## Contact

Questions or feedback? Reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).
