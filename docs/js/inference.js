/*
 * LeafMedic — on-device inference with ONNX Runtime Web.
 * Model: MobileNet, uint8-quantized, input [1,300,300,3] RGB, output [1,16] uint8.
 * Mirrors the preprocessing of the Python app (plain resize, no normalization).
 *
 * Two runtime bundles are vendored and one is chosen at load time:
 *   - WebGPU  (ort.webgpu.min.js + asyncify wasm, 5.6 MB gzipped) when the
 *     browser exposes a working GPU adapter. Measured ~4.7x faster than WASM
 *     on this model, with bit-identical outputs.
 *   - WASM    (ort.wasm.min.js, 3.2 MB gzipped) everywhere else.
 * Loading them conditionally means non-WebGPU browsers never download the
 * larger binary.
 */
'use strict';

const LeafModel = (() => {
  const MODEL_URL = 'model/leafmedic.onnx';
  const INPUT_SIZE = 300;

  let session = null;
  let labels = [];
  let treatments = {};
  let backend = 'unknown';
  let lastInferenceMs = 0;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = src;
      el.onload = resolve;
      el.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(el);
    });
  }

  /* WebGPU is only worth its larger runtime if an adapter actually exists —
   * navigator.gpu can be present while requestAdapter() returns null (no
   * compatible hardware, or the GPU is blocklisted). */
  async function webgpuAvailable() {
    if (!navigator.gpu || typeof navigator.gpu.requestAdapter !== 'function') return false;
    try {
      return !!(await navigator.gpu.requestAdapter());
    } catch {
      return false;
    }
  }

  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
    const total = Number(res.headers.get('Content-Length')) || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress) onProgress(received / total);
    }
    const out = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { out.set(c, offset); offset += c.length; }
    return out;
  }

  /* Care guidance is per-language data. English is the source of truth and the
   * fallback: a translation that fails to load must not break diagnosis. */
  async function fetchTreatments(lang) {
    const url = lang && lang !== 'en' ? `data/treatments.${lang}.json` : 'data/treatments.json';
    try {
      const res = await fetch(url);
      if (res.ok) return await res.json();
      console.warn(`[LeafMedic] no knowledge base for "${lang}", falling back to English`);
    } catch (err) {
      console.warn(`[LeafMedic] failed to load ${url}:`, err.message);
    }
    return (await fetch('data/treatments.json')).json();
  }

  /* Swap the knowledge base when the user changes language, without
   * re-downloading or re-creating the inference session. */
  async function setLanguage(lang) {
    treatments = await fetchTreatments(lang);
  }

  async function load(onProgress) {
    const wantGPU = await webgpuAvailable();
    await loadScript(wantGPU ? 'vendor/ort.webgpu.min.js' : 'vendor/ort.wasm.min.js');

    ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    // The service worker injects COOP/COEP headers, so repeat visits are
    // cross-origin isolated and can use multi-threaded WASM. First visit
    // (no controlling worker yet) falls back to a single thread.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    const lang = typeof I18n !== 'undefined' ? I18n.getLang() : 'en';
    const [modelBytes, labelsRes, treatRes] = await Promise.all([
      fetchWithProgress(MODEL_URL, onProgress),
      fetch('data/labels.json'),
      fetchTreatments(lang),
    ]);
    labels = await labelsRes.json();
    treatments = treatRes;

    // Try the preferred provider, then fall back rather than failing outright:
    // a browser can advertise an adapter and still refuse to create a device.
    const providers = wantGPU ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError = null;
    for (const ep of providers) {
      try {
        session = await ort.InferenceSession.create(modelBytes.buffer, {
          executionProviders: [ep],
        });
        backend = ep;
        lastError = null;
        break;
      } catch (err) {
        lastError = err;
        console.warn(`[LeafMedic] ${ep} backend unavailable:`, err.message);
      }
    }
    if (!session) throw lastError || new Error('No usable inference backend');

    // Warm-up runs so the first real diagnosis never pays allocation cost.
    // WebGPU compiles shaders lazily, and a single pass leaves some of that
    // cost for the first real image (~300 ms vs a ~14 ms steady state), so
    // warm up a few times.
    const warm = new ort.Tensor(
      'uint8', new Uint8Array(INPUT_SIZE * INPUT_SIZE * 3), [1, INPUT_SIZE, INPUT_SIZE, 3]
    );
    for (let i = 0; i < 3; i++) {
      await session.run({ [session.inputNames[0]]: warm });
    }
  }

  function isReady() { return session !== null; }
  function getLabels() { return labels; }
  function getTreatment(label) { return treatments[label] || null; }
  function getAllTreatments() { return treatments; }
  function getBackend() { return backend; }
  function getLastInferenceMs() { return lastInferenceMs; }
  function getInputSize() { return INPUT_SIZE; }

  /* Draw an image-like source onto the 300x300 work canvas and return both the
   * model input tensor and the quality metrics, computed in a single pass. */
  function toInputTensor(source, sw, sh) {
    const canvas = document.getElementById('work-canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

    const rgb = new Uint8Array(INPUT_SIZE * INPUT_SIZE * 3);
    for (let i = 0, j = 0; i < data.length; i += 4) {
      rgb[j++] = data[i];
      rgb[j++] = data[i + 1];
      rgb[j++] = data[i + 2];
    }
    const metrics = LeafQuality.scanPixels(data, INPUT_SIZE, INPUT_SIZE);
    return {
      tensor: new ort.Tensor('uint8', rgb, [1, INPUT_SIZE, INPUT_SIZE, 3]),
      rgb,
      metrics,
    };
  }

  /* uint8 affine-quantized scores -> probabilities in [0, 1]. */
  function toProbabilities(output) {
    return output.data instanceof Uint8Array
      ? Float32Array.from(output.data, (v) => v / 255)
      : Float32Array.from(output.data);
  }

  async function runTensor(tensor) {
    const t0 = performance.now();
    const output = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
    lastInferenceMs = performance.now() - t0;
    return toProbabilities(output);
  }

  function rank(probsArr, k) {
    return probsArr
      .map((confidence, i) => ({
        label: labels[i],
        name: (treatments[labels[i]] || {}).common_name
          || labels[i].replace(/___/g, ' — ').replace(/_/g, ' '),
        confidence,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, k);
  }

  /*
   * Classify an image source (HTMLImageElement, HTMLVideoElement, or canvas).
   * Returns the ranked predictions plus the full quality assessment, so the UI
   * can downgrade an unreliable result to "Uncertain".
   */
  async function classify(source, k = 3) {
    if (!session) throw new Error('Model not loaded');
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    const { tensor, rgb, metrics } = toInputTensor(source, sw, sh);

    const probs = await runTensor(tensor);
    const probsArr = Array.from(probs);
    const preds = rank(probsArr, k);

    const quality = LeafQuality.assess(metrics, {
      topConfidence: preds[0] ? preds[0].confidence : 0,
      entropy: LeafQuality.normalizedEntropy(probsArr),
    });

    return {
      preds,
      probs: probsArr,
      rgb,
      quality,
      // Kept flat for backwards compatibility with stored history entries.
      leafScore: quality.leafScore,
      entropy: quality.entropy,
      inferenceMs: lastInferenceMs,
    };
  }

  /*
   * Occlusion sensitivity: slide an opaque patch across the image, re-run
   * inference for each position, and record how far the target class's
   * confidence falls. Regions whose removal hurts the prediction most are the
   * ones the network is relying on.
   *
   * Model-agnostic on purpose — the graph is fully quantized and exposes no
   * gradients or intermediate activations, so Grad-CAM is not an option here.
   *
   * Returns a coarse gridSize x gridSize Float32Array of confidence drops,
   * normalized to [0, 1].
   */
  async function occlusionMap(rgb, classIndex, {
    gridSize = 8, patchValue = 128, onProgress = null,
  } = {}) {
    if (!session) throw new Error('Model not loaded');
    const step = Math.ceil(INPUT_SIZE / gridSize);
    // Patch is 1.5 cells wide so neighbouring probes overlap; a patch exactly
    // one cell wide tends to miss features straddling a cell boundary.
    const patch = Math.round(step * 1.5);

    const base = await runTensor(
      new ort.Tensor('uint8', rgb.slice(), [1, INPUT_SIZE, INPUT_SIZE, 3])
    );
    const baseConfidence = base[classIndex];

    const drops = new Float32Array(gridSize * gridSize);
    const buffer = new Uint8Array(rgb.length);
    let done = 0;

    for (let gy = 0; gy < gridSize; gy++) {
      for (let gx = 0; gx < gridSize; gx++) {
        buffer.set(rgb);
        const x0 = Math.max(0, gx * step - (patch - step) / 2);
        const y0 = Math.max(0, gy * step - (patch - step) / 2);
        const x1 = Math.min(INPUT_SIZE, x0 + patch);
        const y1 = Math.min(INPUT_SIZE, y0 + patch);
        for (let y = y0; y < y1; y++) {
          const rowStart = (y * INPUT_SIZE + x0) * 3;
          buffer.fill(patchValue, rowStart, rowStart + (x1 - x0) * 3);
        }
        const probs = await runTensor(
          new ort.Tensor('uint8', buffer.slice(), [1, INPUT_SIZE, INPUT_SIZE, 3])
        );
        drops[gy * gridSize + gx] = Math.max(0, baseConfidence - probs[classIndex]);
        done++;
        if (onProgress) onProgress(done / (gridSize * gridSize));
      }
    }

    let max = 0;
    for (const d of drops) if (d > max) max = d;
    if (max > 0) for (let i = 0; i < drops.length; i++) drops[i] /= max;

    return { drops, gridSize, baseConfidence };
  }

  return {
    load, isReady, classify, occlusionMap, setLanguage,
    getLabels, getTreatment, getAllTreatments,
    getBackend, getLastInferenceMs, getInputSize,
  };
})();
