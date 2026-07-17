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
    ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated

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

  /* Draw an image-like source onto a 300x300 canvas and return RGB uint8 data. */
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
    return new ort.Tensor('uint8', rgb, [1, INPUT_SIZE, INPUT_SIZE, 3]);
  }

  /*
   * Classify an image source (HTMLImageElement, HTMLVideoElement, or canvas).
   * Returns [{label, name, confidence}] sorted by confidence, top `k`.
   */
  async function classify(source, k = 3) {
    if (!session) throw new Error('Model not loaded');
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    const tensor = toInputTensor(source, sw, sh);
    const feeds = { [session.inputNames[0]]: tensor };
    const output = (await session.run(feeds))[session.outputNames[0]];

    let probs;
    if (output.data instanceof Uint8Array) {
      probs = Float32Array.from(output.data, (v) => v / 255);
    } else {
      probs = Float32Array.from(output.data);
    }
    return Array.from(probs)
      .map((confidence, i) => ({
        label: labels[i],
        name: (treatments[labels[i]] || {}).common_name || labels[i].replace(/___/g, ' — ').replace(/_/g, ' '),
        confidence,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, k);
  }

  return { load, isReady, classify, getLabels, getTreatment, getAllTreatments };
})();
