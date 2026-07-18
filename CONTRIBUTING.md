# Contributing to LeafMedic

Thanks for your interest! This document covers the development workflow, architecture internals, and troubleshooting — everything you need to work on the project. For the overview and quick start, see the [README](README.md).

## Contributing workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

Ideas that would be especially welcome:

- Running the [`training/`](training/README.md) pipeline to produce a 38-class model that fixes the healthy-leaf weakness
- Additional languages — add a string table to `docs/js/i18n.js` and a `treatments.<lang>.json`
- A spatial-coherence signal that catches the uniform-noise gap in the OOD guard (see [Known gap](#a-known-gap-stated-plainly))
- Better treatment recommendations
- UI improvements in either app

## Development workflow

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

## Project structure

```
LeafMedic/
├── main.py                      # Desktop app entry point
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
├── .github/workflows/ci.yml     # CI: lint/types, Python tests, browser smoke test
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

## Core components

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

## Model specification

| Property | Value |
|---|---|
| Architecture | MobileNet (depthwise-separable CNN, edge-optimized) |
| Input tensor | `[1, 300, 300, 3]`, `uint8`, NHWC, raw RGB |
| Output tensor | `[1, 16]`, `uint8` affine-quantized scores |
| Quantization | Full-integer (weights **and** activations), ~4× smaller than float32 |
| Artifact size | 11 MB (`.tflite` and `.onnx` alike) |
| Preprocessing | Bilinear resize only — **no mean/std normalization** |
| Confidence floor | 0.30, applied after dequantization |

Because the network is fully integer-quantized, activations stay in `uint8` end to end and outputs are rescaled to probabilities with a single `v / 255` division.

### Cross-runtime parity

Shipping one network through two runtimes only works if preprocessing matches bit-for-bit. Two decisions enforce that:

- **Identical resize semantics.** `cv2.resize` (desktop) and `ctx.drawImage` onto a 300×300 canvas (browser) both perform a plain bilinear resize with no aspect-ratio padding, so the tensors agree.
- **No normalization anywhere.** The quantized graph bakes the scale/zero-point into the model itself, so neither runtime applies ImageNet statistics. `ml_module.py` keeps a float32 normalization branch purely as a fallback for swapping in a non-quantized model later.

Parity is not assumed — `tests/test_parity.py` asserts TFLite and ONNX produce byte-identical outputs, and `tests/test_treatments.py` asserts the desktop and web label/knowledge-base files are byte-identical, so a change to one that isn't mirrored fails CI.

## The trust guards

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

### A known gap, stated plainly

Uniform RGB noise defeats all three heuristics. It contains genuinely leaf-coloured pixels and is extremely sharp, so it passes. A Laplacian *ceiling* cannot separate it from a detailed real photograph (real leaves here reach ~19,900 variance against noise's ~48,800); rejecting it properly needs a spatial-coherence signal. `test_uniform_noise_is_a_known_gap` documents this rather than hiding it.

## Explainability: occlusion sensitivity

Confidence bars say *how sure* the model is, not *why*. The **"Why this diagnosis?"** button answers the second question with occlusion sensitivity: slide an opaque patch across the image, re-run inference at each position, and measure how far the predicted class's confidence falls. Regions whose removal hurts the prediction most are the ones the network relied on.

The shipped graph is fully integer-quantized and exposes neither gradients nor intermediate activations, so Grad-CAM is not an option. Occlusion needs nothing but repeated forward passes — 64 extra inferences on an 8×8 grid, with patches overlapping at 1.5 cells so features straddling a cell boundary are not missed. On WebGPU that is roughly a second.

## Browser runtime engineering

The demo is a static site on GitHub Pages, which imposes real constraints that shaped the implementation:

- **WebGPU when it pays for itself.** ONNX Runtime ships separate bundles per backend, and the WebGPU-capable one needs a 5.6 MB gzipped WASM binary against 3.2 MB for the WASM-only build. Rather than guess, both were benchmarked on this model: WebGPU runs at a **14.3 ms median against 63.7 ms** for multi-threaded WASM — 4.5x, with a bit-identical output checksum. Worth 2.4 MB, so the demo probes for a real GPU adapter (`navigator.gpu` can exist while `requestAdapter()` returns null) and injects the matching bundle at load time. Browsers without WebGPU never download the larger binary, and a browser that advertises an adapter but fails to create a device falls back to WASM instead of erroring.
- **Multi-threaded WASM without server headers.** SIMD-threaded ONNX Runtime requires `crossOriginIsolated`, which requires COOP/COEP response headers — and GitHub Pages cannot send custom headers. The service worker resolves this by intercepting same-origin responses and re-issuing them with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. First visit runs single-threaded; once the worker is in control, inference scales to `min(4, hardwareConcurrency)` threads. Every asset is same-origin, so `require-corp` breaks nothing.
- **Streamed download with real progress.** The 11 MB model is fetched through a `ReadableStream` reader that tracks `Content-Length` to drive an accurate progress bar, instead of a spinner that lies until the whole buffer lands.
- **Tiered caching.** The service worker is network-first for the app shell (so updates reach returning visitors immediately) and cache-first for immutable heavy assets — model, WASM binaries, fonts, and photos.
- **Zero third-party runtime dependencies.** ONNX Runtime Web and both typefaces are self-hosted; the page makes no CDN, analytics, or telemetry requests, which is what makes the privacy claim verifiable rather than promotional.
- **Warm-up sized to the backend.** WebGPU compiles shaders lazily, so a single warm-up pass leaves ~300 ms of that cost for the user's first real photo. Three passes on a zero tensor at load time move it off the critical path.

## Testing details

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

## Training your own model

The shipped model is pretrained; [`training/`](training/README.md) is the pipeline for replacing it. It fine-tunes an ImageNet backbone on PlantVillage's 38 classes, quantizes to full-integer uint8 with a real calibration set, exports both `.tflite` and `.onnx`, and verifies the two agree before you deploy them.

```bash
pip install -e '.[train]'
python3 training/download_dataset.py --per-class 200
python3 training/train.py --epochs 15 --fine-tune-epochs 8
python3 training/evaluate.py --model training/runs/<run>/model_int8.tflite
```

There is headroom waiting: `data/treatments.json` already covers **all 38 PlantVillage classes**, so the gap closes with no new care guidance to write. See [`training/README.md`](training/README.md) for the design decisions — why preprocessing lives inside the graph, why quantization calibration needs real images, and what to check when swapping a new model in.

**One caveat, measured rather than assumed:** the pipeline runs end to end, but the TFLite→ONNX conversion is *not* bit-exact for models it produces, unlike the shipped one. MobileNetV2 lands within 6 quantization steps of 255, MobileNetV3-Small within 18. That is small enough not to change a confident diagnosis and large enough to matter before promising the browser and desktop apps agree exactly. [`training/README.md`](training/README.md#open-issue-onnx-conversion-is-not-bit-exact-for-new-models) documents the numbers and what to check before deploying.

## Raspberry Pi setup

**Hardware:** Raspberry Pi 4 Model B (4 GB recommended) · Arducam 5 MP OV5647 camera module · monitor · 5V 3A USB-C supply

**Camera connection:**

1. Power off the Raspberry Pi
2. Locate the camera connector (between HDMI ports and audio jack)
3. Pull up the plastic clip gently
4. Insert ribbon cable (**blue side facing audio jack**, contacts facing HDMI)
5. Push the clip down to secure
6. Power on

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

## Capture tips

| Factor | Recommendation |
|---|---|
| Distance | 20–30 cm from the leaf |
| Lighting | Natural daylight or bright LED; avoid shadows |
| Framing | A single in-focus leaf filling most of the frame |
| Angle | Perpendicular to the leaf surface |
| Background | Plain and contrasting |

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

## Acknowledgments

- **AgriPredict** — Disease Classification TFLite model on Kaggle
- **PlantVillage** — 54,000+ image dataset for future model training
- **TensorFlow** & **ONNX Runtime** — inference runtimes
- **Raspberry Pi Foundation** — hardware platform and Picamera2
- **SunFounder** — the electronics kit where this project began
