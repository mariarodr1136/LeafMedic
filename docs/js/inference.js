/*
 * LeafMedic — on-device inference with ONNX Runtime Web.
 * Model: MobileNetV2, uint8-quantized, input [1,300,300,3] RGB, output [1,16] uint8.
 * Mirrors the preprocessing of the Python app (plain resize, no normalization).
 */
'use strict';

const LeafModel = (() => {
  const MODEL_URL = 'model/leafmedic.onnx';
  const INPUT_SIZE = 300;

  let session = null;
  let labels = [];
  let treatments = {};

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

  async function load(onProgress) {
    ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    // The service worker injects COOP/COEP headers, so repeat visits are
    // cross-origin isolated and can use multi-threaded WASM. First visit
    // (no controlling worker yet) falls back to a single thread.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    const [modelBytes, labelsRes, treatRes] = await Promise.all([
      fetchWithProgress(MODEL_URL, onProgress),
      fetch('data/labels.json'),
      fetch('data/treatments.json'),
    ]);
    labels = await labelsRes.json();
    treatments = await treatRes.json();
    session = await ort.InferenceSession.create(modelBytes.buffer, {
      executionProviders: ['wasm'],
    });
  }

  function isReady() { return session !== null; }
  function getLabels() { return labels; }
  function getTreatment(label) { return treatments[label] || null; }
  function getAllTreatments() { return treatments; }

  /* Draw an image-like source onto a 300x300 canvas and return RGB uint8 data
   * plus a "leaf score": the fraction of vegetation-colored pixels (lenient
   * yellow-to-green hue check, so brown/diseased leaves still pass). */
  function toInputTensor(source, sw, sh) {
    const canvas = document.getElementById('work-canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const rgb = new Uint8Array(INPUT_SIZE * INPUT_SIZE * 3);
    let vegetation = 0;
    for (let i = 0, j = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      rgb[j++] = r;
      rgb[j++] = g;
      rgb[j++] = b;
      // Vegetation-ish: green channel not dominated by blue, and either
      // green-leaning (healthy) or warm yellow/brown (diseased tissue).
      if (g >= b && (g >= r - 20) && g > 40) vegetation++;
    }
    const leafScore = vegetation / (INPUT_SIZE * INPUT_SIZE);
    return {
      tensor: new ort.Tensor('uint8', rgb, [1, INPUT_SIZE, INPUT_SIZE, 3]),
      leafScore,
    };
  }

  /* Shannon entropy of the probability vector, normalized to [0, 1].
   * ~0 = one confident class, ~1 = uniform (model has no idea). */
  function normalizedEntropy(probs) {
    const sum = probs.reduce((a, p) => a + p, 0) || 1;
    let h = 0;
    for (const p of probs) {
      const q = p / sum;
      if (q > 0) h -= q * Math.log(q);
    }
    return h / Math.log(probs.length);
  }

  /*
   * Classify an image source (HTMLImageElement, HTMLVideoElement, or canvas).
   * Returns { preds: [{label, name, confidence}] top-k by confidence,
   *           leafScore, entropy } for out-of-distribution detection.
   */
  async function classify(source, k = 3) {
    if (!session) throw new Error('Model not loaded');
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    const { tensor, leafScore } = toInputTensor(source, sw, sh);
    const feeds = { [session.inputNames[0]]: tensor };
    const output = (await session.run(feeds))[session.outputNames[0]];

    let probs;
    if (output.data instanceof Uint8Array) {
      probs = Float32Array.from(output.data, (v) => v / 255);
    } else {
      probs = Float32Array.from(output.data);
    }
    const probsArr = Array.from(probs);
    const preds = probsArr
      .map((confidence, i) => ({
        label: labels[i],
        name: (treatments[labels[i]] || {}).common_name || labels[i].replace(/___/g, ' — ').replace(/_/g, ' '),
        confidence,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, k);
    return { preds, leafScore, entropy: normalizedEntropy(probsArr) };
  }

  return { load, isReady, classify, getLabels, getTreatment, getAllTreatments };
})();
