/* LeafMedic web app — UI, camera, upload, results, history, library. */
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);

  const SAMPLES = [
    { file: 'corn_rust_1.jpg', caption: 'Corn leaf' },
    { file: 'corn_rust_2.jpg', caption: 'Corn leaf' },
    { file: 'corn_rust_3.jpg', caption: 'Corn leaf' },
    { file: 'tomato_bacterial_spot_1.jpg', caption: 'Tomato leaf' },
    { file: 'tomato_bacterial_spot_2.jpg', caption: 'Tomato leaf' },
    { file: 'tomato_bacterial_spot_3.jpg', caption: 'Tomato leaf' },
    { file: 'tomato_leaf_mold_1.jpg', caption: 'Tomato leaf' },
    { file: 'tomato_leaf_mold_2.jpg', caption: 'Tomato leaf' },
    { file: 'tomato_leaf_mold_3.jpg', caption: 'Tomato leaf' },
  ];
  const HISTORY_KEY = 'leafmedic-history';
  const HISTORY_MAX = 12;
  const CONFIDENCE_FLOOR = 0.3;
  // Photos accepted per combined diagnosis. Each one costs a full inference
  // pass, so the cap keeps the slowest backend (single-thread WASM) responsive.
  const MULTI_MAX = 5;

  // Library card photos. Sources and licenses: docs/img/CREDITS.md
  // (Corn lethal necrosis has no openly-licensed photo yet, so it is left out
  // of the library grid — the model can still diagnose it.)
  const LIBRARY_IMAGES = {
    'Tomato___healthy': 'img/library/tomato-healthy.jpg',
    'Tomato___Septoria_leaf_spot': 'img/library/tomato-septoria-leaf-spot.jpg',
    'Tomato___Bacterial_spot': 'img/library/tomato-bacterial-spot.jpg',
    'Tomato___Late_blight': 'img/library/tomato-late-blight.jpg',
    'Tomato___Spider_mites': 'img/library/tomato-spider-mites.jpg',
    'Tomato___Leaf_Mold': 'img/library/tomato-leaf-mold.jpg',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'img/library/tomato-yellow-leaf-curl-virus.jpg',
    'Corn_(maize)___Common_rust_': 'img/library/corn-common-rust.jpg',
    'Corn_(maize)___healthy': 'img/library/corn-healthy.jpg',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'img/library/corn-gray-leaf-spot.jpg',
    'Soybean___healthy': 'img/library/soybean-healthy.jpg',
    'Soybean___Downy_Mildew': 'img/library/soybean-downy-mildew.jpg',
    'Soybean___Frogeye_Leaf_Spot': 'img/library/soybean-frogeye-leaf-spot.jpg',
    'Cabbage___healthy': 'img/library/cabbage-healthy.jpg',
    'Cabbage___Black_Rot': 'img/library/cabbage-black-rot.jpg',
  };

  const VIEWS = ['detect', 'diagnose', 'library', 'about'];
  const t = (key) => I18n.t(key);

  /* ---------- View tabs ---------- */
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  });
  $('brand-link').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  $('crumb-home').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  $('footer-brand-link').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  // Footer links and the home page's "diagnose now" CTA jump between views
  // the same way.
  document.querySelectorAll('[data-goto]').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      showView(el.dataset.goto);
    });
  });
  let currentView = 'detect';
  function showView(name) {
    currentView = VIEWS.includes(name) ? name : 'detect';
    document.querySelectorAll('.tab').forEach((tab) => {
      const active = tab.dataset.view === currentView;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${currentView}`));
    renderHero();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (currentView !== 'diagnose') stopCamera();
  }

  function renderHero() {
    $('crumb-current').textContent = t(`nav.${currentView}`);
    $('hero-title').textContent = t(`hero.${currentView}.title`);
    $('hero-copy').textContent = t(`hero.${currentView}.copy`);
    $('hero-cta').hidden = currentView === 'about' || currentView === 'diagnose';
  }

  /* ---------- Hero CTA ---------- */
  $('hero-cta').addEventListener('click', () => {
    showView('diagnose');
    $('detect-tool').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  /* ---------- Model loading ---------- */
  const banner = $('model-banner');
  const progressFill = $('model-progress');
  async function boot() {
    try {
      await LeafModel.load((frac) => {
        progressFill.style.width = `${Math.round(frac * 100)}%`;
      });
      banner.classList.add('done');
      setTimeout(() => banner.remove(), 600);
      renderLibrary();
      enableInputs();
      console.log(`[LeafMedic] inference backend: ${LeafModel.getBackend()}`);
      if (BENCH_MODE) document.body.classList.add('bench-mode');
    } catch (err) {
      console.error(err);
      $('model-status-title').textContent = t('model.failed');
      $('model-status-detail').textContent = `${err.message} — check your connection and refresh.`;
      banner.classList.add('error');
    }
  }

  /* ---------- Language switcher ---------- */
  function buildLanguageSwitcher() {
    const select = $('lang-select');
    I18n.supported().forEach((lang) => {
      const opt = document.createElement('option');
      opt.value = lang;
      opt.textContent = lang.toUpperCase();
      select.appendChild(opt);
    });
    select.value = I18n.getLang();
    select.addEventListener('change', async () => {
      I18n.setLang(select.value);
      // Care guidance is data, not UI chrome: reload the matching knowledge base
      // and repaint anything already showing translated content.
      await LeafModel.setLanguage(select.value);
      renderLibrary();
      renderHistory();
      if (lastRendered) {
        renderResults(lastRendered.preds, lastRendered.thumbUrl,
          lastRendered.quality, lastRendered.batch);
      }
    });
  }

  let inputsEnabled = false;
  function enableInputs() { inputsEnabled = true; document.body.classList.add('model-ready'); }

  /* ---------- Input mode tabs ---------- */
  document.querySelectorAll('.input-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.input-tab').forEach((t) => {
        const active = t === tab;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', String(active));
      });
      document.querySelectorAll('.input-mode').forEach((m) => {
        m.classList.toggle('active', m.id === `mode-${tab.dataset.mode}`);
      });
      if (tab.dataset.mode !== 'camera') stopCamera();
    });
  });

  /* ---------- Samples ---------- */
  const sampleGrid = $('sample-grid');
  SAMPLES.forEach(({ file, caption }) => {
    const btn = document.createElement('button');
    btn.className = 'sample-item';
    btn.innerHTML = `<img src="samples/${file}" alt="${caption} sample" loading="lazy"><span>${caption}</span>`;
    btn.addEventListener('click', () => {
      const img = new Image();
      img.onload = () => analyze(img);
      img.src = `samples/${file}`;
    });
    sampleGrid.appendChild(btn);
  });

  /* ---------- Upload: click / drag / paste ---------- */
  const dropzone = $('dropzone');
  const fileInput = $('file-input');
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) loadFiles([...fileInput.files]); fileInput.value = ''; });
  ['dragover', 'dragenter'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragging');
  }));
  dropzone.addEventListener('drop', (e) => {
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'));
    if (files.length) loadFiles(files);
  });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) loadFiles([item.getAsFile()]);
  });
  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error(`Could not read ${file.name}`)); };
      img.src = url;
    });
  }
  // Several files at once become one combined diagnosis of the same plant.
  async function loadFiles(files) {
    try {
      const images = await Promise.all(files.slice(0, MULTI_MAX).map(loadImage));
      if (images.length === 1) analyze(images[0]);
      else analyzeBatch(images);
    } catch (err) {
      console.error(err);
      alert(err.message);
    }
  }

  /* ---------- Camera ---------- */
  const video = $('camera-video');
  const cameraPlaceholder = $('camera-placeholder');
  const cameraControls = $('camera-controls');
  let stream = null;
  let facing = 'environment';

  $('camera-start').addEventListener('click', startCamera);
  $('camera-flip').addEventListener('click', () => {
    facing = facing === 'environment' ? 'user' : 'environment';
    startCamera();
  });
  $('camera-capture').addEventListener('click', () => {
    if (video.readyState >= 2) analyze(video);
  });

  async function startCamera() {
    stopCamera();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      video.classList.toggle('mirrored', facing === 'user');
      cameraPlaceholder.hidden = true;
      cameraControls.hidden = false;
    } catch (err) {
      $('camera-message').textContent =
        err.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in your browser settings and try again.'
          : `Could not start the camera (${err.name}). You can still upload a photo instead.`;
      cameraPlaceholder.hidden = false;
      cameraControls.hidden = true;
    }
  }
  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      video.srcObject = null;
      cameraPlaceholder.hidden = false;
      cameraControls.hidden = true;
    }
  }

  /* ---------- Analysis & results ---------- */
  const resultsPanel = $('results-panel');

  // Retained so "Why this diagnosis?" can re-run inference on the exact tensor
  // that produced the current result.
  let lastAnalysis = null;

  // Out-of-distribution guard: top confidence too low, prediction spread
  // too flat, or the image barely contains vegetation-colored pixels.
  function unreliableResult(preds, quality) {
    return !!(quality && (quality.notLeaf || quality.uncertain)) ||
      preds[0].confidence < CONFIDENCE_FLOOR;
  }

  async function analyze(source) {
    if (!inputsEnabled) return;
    resultsPanel.classList.add('analyzing');
    try {
      const thumb = makeThumbnail(source, 320);
      const result = await LeafModel.classify(source, 3);
      lastAnalysis = { ...result, thumb, batch: null };
      renderResults(result.preds, thumb, result.quality);
      // An unreliable result is stored flagged, so history shows it as
      // "Uncertain" instead of presenting the class name as a diagnosis.
      pushHistory(result.preds, makeThumbnail(source, 96),
        unreliableResult(result.preds, result.quality));
      if (BENCH_MODE) reportBench(result);
    } catch (err) {
      console.error(err);
      alert(`Analysis failed: ${err.message}`);
    } finally {
      resultsPanel.classList.remove('analyzing');
    }
  }

  /* Two or more photos of the same plant: average the predictions (see
   * LeafModel.classifyBatch), represent the result with the sharpest photo. */
  async function analyzeBatch(sources) {
    if (!inputsEnabled) return;
    resultsPanel.classList.add('analyzing');
    try {
      const result = await LeafModel.classifyBatch(sources, 3);
      const best = sources[result.bestIndex];
      const thumb = makeThumbnail(best, 320);
      const batch = { used: result.imagesUsed, total: result.imagesTotal };
      lastAnalysis = { ...result, thumb, batch };
      renderResults(result.preds, thumb, result.quality, batch);
      pushHistory(result.preds, makeThumbnail(best, 96),
        unreliableResult(result.preds, result.quality), result.imagesUsed);
      if (BENCH_MODE) reportBench(result);
    } catch (err) {
      console.error(err);
      alert(`Analysis failed: ${err.message}`);
    } finally {
      resultsPanel.classList.remove('analyzing');
    }
  }

  function makeThumbnail(source, size) {
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    const scale = size / Math.max(sw, sh);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(sw * scale);
    canvas.height = Math.round(sh * scale);
    canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  }

  // What the results panel currently shows — the report builder reads this,
  // so a report can be saved even for a history replay (which has no
  // lastAnalysis pixel buffer).
  let lastRendered = null;

  function renderResults(rawPreds, thumbUrl, quality, batch = null) {
    // Re-resolve display names from labels on every render so a language
    // change repaints existing results instead of keeping the names captured
    // at analysis time.
    const preds = rawPreds.map((p) => {
      const entry = LeafModel.getTreatment(p.label);
      return { ...p, name: (entry && entry.common_name) || p.name };
    });
    const top = preds[0];
    const info = LeafModel.getTreatment(top.label);
    const healthy = /healthy/i.test(top.label);
    const notLeaf = !!(quality && quality.notLeaf);
    const lowConfidence = unreliableResult(preds, quality);
    lastRendered = { preds, thumbUrl, quality, batch, lowConfidence, healthy };

    $('results-empty').hidden = true;
    $('results-content').hidden = false;
    $('analyzed-img').src = thumbUrl;

    $('low-confidence-note').innerHTML = notLeaf ? t('results.notLeaf') : t('results.lowConfidence');
    $('low-confidence-note').hidden = !lowConfidence;

    // Combined-diagnosis note: how many photos went into this result, and
    // whether any were rejected as not-a-leaf and left out of the average.
    const multiNote = $('multi-note');
    if (batch && batch.used > 1) {
      let note = t('results.multiNote').replace('{n}', batch.used);
      if (batch.total > batch.used) {
        note += ` ${t('results.multiSkipped').replace('{m}', batch.total - batch.used)}`;
      }
      multiNote.textContent = note;
      multiNote.hidden = false;
    } else {
      multiNote.hidden = true;
    }

    // Capture-quality problems are actionable in a way a low score is not:
    // the fix is "retake the photo like this", so they get their own note.
    const captureNote = $('capture-note');
    const captureIssues = [];
    if (quality && quality.blurry) captureIssues.push(t('quality.blurry'));
    if (quality && quality.tooDark) captureIssues.push(t('quality.dark'));
    if (quality && quality.tooBright) captureIssues.push(t('quality.bright'));
    captureNote.innerHTML = captureIssues.map((m) => `<span>${m}</span>`).join('');
    captureNote.hidden = captureIssues.length === 0;

    const badge = $('diagnosis-badge');
    if (lowConfidence) {
      badge.textContent = t('results.uncertain');
      badge.className = 'diagnosis-badge uncertain';
    } else if (healthy) {
      badge.textContent = t('results.healthy');
      badge.className = 'diagnosis-badge healthy';
    } else {
      badge.textContent = `${info?.severity || 'medium'} ${t('results.severitySuffix')}`;
      badge.className = `diagnosis-badge severity-${info?.severity || 'medium'}`;
    }
    // An unreliable result shows no diagnosis at all: naming a class with a
    // confidence next to a warning still reads as an answer.
    $('diagnosis-name').textContent = lowConfidence ? t('results.noDiagnosis') : top.name;
    $('diagnosis-sub').textContent = lowConfidence ? '' :
      `${(top.confidence * 100).toFixed(1)}% ${t('results.confidenceSuffix')}`;

    // Explanation is only offered for a result worth explaining, and only when
    // the pixel buffer for this analysis is still around (not a history replay).
    const explainable = !!(lastAnalysis && lastAnalysis.rgb) && !lowConfidence;
    $('explain-block').hidden = !explainable;
    $('explain-result').hidden = true;
    $('explain-btn').disabled = false;

    $('confidence-title').hidden = lowConfidence;
    const bars = $('pred-bars');
    bars.innerHTML = '';
    const barPreds = lowConfidence ? [] : preds;
    barPreds.forEach(({ name, confidence, label }) => {
      const pct = (confidence * 100).toFixed(1);
      const isHealthy = /healthy/i.test(label);
      const row = document.createElement('div');
      row.className = 'pred-row';
      row.innerHTML = `
        <span class="pred-name">${name}</span>
        <div class="pred-track"><div class="pred-fill ${isHealthy ? 'healthy' : ''}" style="width:0%"></div></div>
        <span class="pred-pct">${pct}%</span>`;
      bars.appendChild(row);
      requestAnimationFrame(() =>
        requestAnimationFrame(() => { row.querySelector('.pred-fill').style.width = `${pct}%`; })
      );
    });

    $('treatment-card').innerHTML = lowConfidence || healthy || !info ? (!lowConfidence && healthy ? `
      <div class="healthy-note">
        <strong>🌿 ${t('results.healthyTitle')}</strong>
        <p>${t('results.healthyCopy')}</p>
      </div>` : '') : treatmentHTML(info);

    // A report is only offered for a result worth keeping — an Uncertain
    // verdict has nothing to print.
    $('report-block').hidden = lowConfidence;
  }

  function treatmentHTML(info) {
    const list = (items) => items.map((i) => `<li>${i}</li>`).join('');
    return `
      <h3 class="section-title">${t('treatment.about')} ${info.common_name}</h3>
      <p class="treatment-desc">${info.description}</p>
      <div class="treatment-cols">
        <div class="treatment-col">
          <h4><span aria-hidden="true">🔍</span> ${t('treatment.symptoms')}</h4>
          <ul>${list(info.symptoms || [])}</ul>
        </div>
        <div class="treatment-col">
          <h4><span aria-hidden="true">💊</span> ${t('treatment.treatment')}</h4>
          <ul>${list(info.treatments || [])}</ul>
        </div>
        <div class="treatment-col">
          <h4><span aria-hidden="true">🛡️</span> ${t('treatment.prevention')}</h4>
          <ul>${list(info.prevention || [])}</ul>
        </div>
      </div>`;
  }

  /* ---------- Explainability: occlusion sensitivity ---------- */
  // Slides a patch over the image, re-runs inference for each position, and
  // paints the confidence drop as a heatmap. The model is a fully-quantized
  // black box with no gradients, so this is the practical alternative to
  // Grad-CAM: it needs nothing but repeated forward passes.
  const GRID = 8;

  $('explain-btn').addEventListener('click', async () => {
    if (!lastAnalysis || !lastAnalysis.rgb) return;
    const btn = $('explain-btn');
    const hint = $('explain-hint');
    btn.disabled = true;
    hint.textContent = t('results.explaining');

    try {
      const top = lastAnalysis.preds[0];
      const classIndex = LeafModel.getLabels().indexOf(top.label);
      const { drops } = await LeafModel.occlusionMap(lastAnalysis.rgb, classIndex, {
        gridSize: GRID,
        onProgress: (frac) => { hint.textContent = `${t('results.explaining')} ${Math.round(frac * 100)}%`; },
      });
      drawHeatmap(lastAnalysis.rgb, drops, GRID);
      $('explain-caption').textContent = t('results.explainCaption');
      $('explain-result').hidden = false;
    } catch (err) {
      console.error(err);
      hint.textContent = `Could not compute the explanation: ${err.message}`;
    } finally {
      btn.disabled = false;
      if (!$('explain-result').hidden) hint.textContent = t('results.explainHint');
    }
  });

  /* Composite the leaf with a warm overlay whose alpha follows the importance
   * grid, upscaled smoothly so the coarse probe grid reads as a soft heatmap. */
  function drawHeatmap(rgb, drops, gridSize) {
    const size = LeafModel.getInputSize();
    const canvas = $('explain-canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    const base = ctx.createImageData(size, size);
    for (let p = 0, j = 0; p < size * size; p++) {
      base.data[p * 4] = rgb[j++];
      base.data[p * 4 + 1] = rgb[j++];
      base.data[p * 4 + 2] = rgb[j++];
      base.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(base, 0, 0);

    // Draw the grid into a small offscreen canvas, then let the 2D context
    // scale it up with its own smoothing — cheaper and softer than
    // interpolating per pixel in JS.
    const heat = document.createElement('canvas');
    heat.width = gridSize;
    heat.height = gridSize;
    const hctx = heat.getContext('2d');
    const hdata = hctx.createImageData(gridSize, gridSize);
    for (let i = 0; i < drops.length; i++) {
      const v = Math.min(1, Math.max(0, drops[i]));
      hdata.data[i * 4] = 255;
      hdata.data[i * 4 + 1] = Math.round(200 * (1 - v));
      hdata.data[i * 4 + 2] = Math.round(60 * (1 - v));
      hdata.data[i * 4 + 3] = Math.round(200 * v);
    }
    hctx.putImageData(hdata, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(heat, 0, 0, gridSize, gridSize, 0, 0, size, size);
  }

  /* ---------- Printable report ---------- */
  // Fills a print-only page with the current result and opens the browser's
  // print dialog. "Save as PDF" is built into every browser's print dialog,
  // so the export needs no library, works offline, and — like everything
  // else here — never sends the photo anywhere.
  function reportHTML() {
    const { preds, thumbUrl, batch, healthy } = lastRendered;
    const top = preds[0];
    const info = LeafModel.getTreatment(top.label);
    const when = new Date().toLocaleString(I18n.getLang(),
      { dateStyle: 'long', timeStyle: 'short' });
    const badge = healthy
      ? `<span class="diagnosis-badge healthy">${t('results.healthy')}</span>`
      : `<span class="diagnosis-badge severity-${info?.severity || 'medium'}">${info?.severity || 'medium'} ${t('results.severitySuffix')}</span>`;
    const multi = batch && batch.used > 1
      ? `<p class="diagnosis-sub">${t('results.multiNote').replace('{n}', batch.used)}</p>` : '';
    const rows = preds.map(({ name, confidence }) => {
      const pct = (confidence * 100).toFixed(1);
      return `
        <div class="pred-row">
          <span class="pred-name">${name}</span>
          <div class="pred-track"><div class="pred-fill" style="width:${pct}%"></div></div>
          <span class="pred-pct">${pct}%</span>
        </div>`;
    }).join('');

    return `
      <header class="report-head">
        <svg class="brand-leaf" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C7 6 4 10.5 4 15a8 8 0 0 0 16 0c0-4.5-3-9-8-13Z" fill="currentColor" stroke="none"/></svg>
        <span class="report-brand">Leaf<em>Medic</em></span>
        <span class="report-title">${t('report.title')}</span>
      </header>
      <p class="report-meta">${t('report.generated')} ${when} · mariarodr1136.github.io/LeafMedic</p>
      <div class="analyzed-row">
        <img class="analyzed-img" src="${thumbUrl}" alt="">
        <div class="diagnosis">
          ${badge}
          <h2>${top.name}</h2>
          <p class="diagnosis-sub">${(top.confidence * 100).toFixed(1)}% ${t('results.confidenceSuffix')}</p>
          ${multi}
        </div>
      </div>
      <h3 class="section-title">${t('results.confidence')}</h3>
      ${rows}
      ${healthy ? `
      <div class="healthy-note">
        <strong>🌿 ${t('results.healthyTitle')}</strong>
        <p>${t('results.healthyCopy')}</p>
      </div>` : (info ? treatmentHTML(info) : '')}
      <p class="disclaimer">${t('results.disclaimer')}</p>`;
  }

  $('report-btn').addEventListener('click', () => {
    if (!lastRendered || lastRendered.lowConfidence) return;
    $('report-page').innerHTML = reportHTML();
    document.body.classList.add('print-report');
    window.print();
  });
  window.addEventListener('afterprint', () => document.body.classList.remove('print-report'));

  /* ---------- Benchmark mode (?bench) ---------- */
  // Turns the README's quoted latency figures into something a visitor can
  // reproduce on their own hardware.
  const BENCH_MODE = new URLSearchParams(location.search).has('bench');

  function reportBench(result) {
    const stats = {
      backend: LeafModel.getBackend(),
      inferenceMs: Number(result.inferenceMs.toFixed(1)),
      crossOriginIsolated: self.crossOriginIsolated,
      threads: self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1,
      top: result.preds[0].label,
      confidence: Number(result.preds[0].confidence.toFixed(3)),
    };
    console.log('[LeafMedic bench]', stats);
    window.LEAFMEDIC_BENCH = stats;
    let el = $('bench-readout');
    if (!el) {
      el = document.createElement('div');
      el.id = 'bench-readout';
      el.className = 'bench-readout';
      document.body.appendChild(el);
    }
    el.textContent =
      `${stats.backend} · ${stats.inferenceMs} ms · ${stats.threads} thread(s)` +
      `${stats.crossOriginIsolated ? ' · isolated' : ''}`;
  }

  /* Run N inferences on a fixed sample and report the distribution. Exposed on
   * window so it can be driven from the console or a headless test. */
  async function runBenchmark(iterations = 20) {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = `samples/${SAMPLES[0].file}`;
    });
    const times = [];
    for (let i = 0; i < iterations; i++) {
      const r = await LeafModel.classify(img, 1);
      times.push(r.inferenceMs);
    }
    times.sort((a, b) => a - b);
    const summary = {
      backend: LeafModel.getBackend(),
      iterations,
      medianMs: Number(times[Math.floor(times.length / 2)].toFixed(1)),
      minMs: Number(times[0].toFixed(1)),
      maxMs: Number(times[times.length - 1].toFixed(1)),
      threads: self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1,
      crossOriginIsolated: self.crossOriginIsolated,
    };
    console.table(summary);
    window.LEAFMEDIC_BENCH_RESULT = summary;
    return summary;
  }
  window.leafmedicBenchmark = runBenchmark;

  /* ---------- History ---------- */
  function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }
  function pushHistory(preds, thumbUrl, uncertain, count = 1) {
    const top = preds[0];
    const entries = getHistory();
    entries.unshift({
      name: top.name, label: top.label, confidence: top.confidence,
      preds: preds.map(({ name, label, confidence }) => ({ name, label, confidence })),
      uncertain: !!uncertain, count, thumb: thumbUrl, ts: Date.now(), plant: '',
    });
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, HISTORY_MAX)));
    } catch { /* storage full — drop silently */ }
    renderHistory();
  }
  // A tag is opt-in and freeform (e.g. "Balcony tomato #1"), letting the same
  // plant's diagnoses be grouped into a timeline without inventing plant IDs.
  function setHistoryPlantTag(ts, plant) {
    const entries = getHistory();
    const entry = entries.find((e) => e.ts === ts);
    if (!entry) return;
    entry.plant = plant.trim();
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(entries)); } catch { /* storage full — drop silently */ }
    renderHistory();
  }
  // Shared by both the history grid and plant timelines, which both replay
  // a stored entry into the results panel the same way.
  function replayHistoryEntry(entry) {
    const preds = entry.preds || [{ name: entry.name, label: entry.label, confidence: entry.confidence }];
    // A replay has no pixel buffer, so the explanation button stays hidden.
    lastAnalysis = null;
    renderResults(preds, entry.thumb, entry.uncertain ? { uncertain: true } : null,
      entry.count > 1 ? { used: entry.count, total: entry.count } : null);
  }
  function historyDisplayName(entry) {
    // Re-resolve the display name from the label so stored history follows
    // the current language instead of freezing the name at analysis time.
    const info = entry.label ? LeafModel.getTreatment(entry.label) : null;
    return (info && info.common_name) || entry.name;
  }
  function formatHistoryWhen(ts) {
    return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  }
  function renderHistory() {
    const entries = getHistory();
    $('history-block').hidden = entries.length === 0;
    const grid = $('history-grid');
    grid.innerHTML = '';
    entries.forEach((entry) => {
      const { confidence, thumb, ts } = entry;
      const name = historyDisplayName(entry);
      const when = formatHistoryWhen(ts);
      // Combined analyses carry how many photos went into them.
      const photos = entry.count > 1 ? ` · ${entry.count} 📷` : '';
      const wrap = document.createElement('div');
      wrap.className = 'history-item-wrap';

      const item = document.createElement('button');
      item.className = 'history-item';
      item.title = 'Show this result again';
      // An uncertain analysis is listed as such — no class name or confidence,
      // which would read as a diagnosis.
      item.innerHTML = `
        <img src="${thumb}" alt="">
        <div class="history-meta">
          ${entry.uncertain
            ? `<strong>${t('results.uncertain')}</strong>
          <span>${when}${photos}</span>`
            : `<strong>${name}</strong>
          <span>${(confidence * 100).toFixed(0)}% · ${when}${photos}</span>`}
        </div>`;
      item.addEventListener('click', () => replayHistoryEntry(entry));
      wrap.appendChild(item);

      const tagInput = document.createElement('input');
      tagInput.className = 'history-tag-input';
      tagInput.type = 'text';
      tagInput.maxLength = 40;
      tagInput.value = entry.plant || '';
      tagInput.placeholder = t('history.tagPlaceholder');
      tagInput.addEventListener('change', () => setHistoryPlantTag(ts, tagInput.value));
      wrap.appendChild(tagInput);

      grid.appendChild(wrap);
    });
    renderTimelines(entries);
  }
  $('history-clear').addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  /* ---------- Plant timelines ----------
   * Diagnoses tagged with the same plant name, grouped and shown oldest to
   * newest, so a recurring issue (or its recovery) is visible at a glance. */
  function renderTimelines(entries) {
    const byPlant = new Map();
    entries.forEach((entry) => {
      const plant = (entry.plant || '').trim();
      if (!plant) return;
      if (!byPlant.has(plant)) byPlant.set(plant, []);
      byPlant.get(plant).push(entry);
    });

    const block = $('timelines-block');
    const list = $('timelines-list');
    list.innerHTML = '';
    const groups = [...byPlant.entries()].filter(([, group]) => group.length > 1);
    block.hidden = groups.length === 0;
    if (groups.length === 0) return;

    groups.forEach(([plant, group]) => {
      const chronological = [...group].sort((a, b) => a.ts - b.ts);
      const card = document.createElement('div');
      card.className = 'timeline-card';
      const header = document.createElement('div');
      header.className = 'timeline-head';
      header.innerHTML = `<strong>${plant}</strong><span>${t('timelines.count').replace('{n}', chronological.length)}</span>`;
      card.appendChild(header);

      const strip = document.createElement('div');
      strip.className = 'timeline-strip';
      chronological.forEach((entry) => {
        const item = document.createElement('button');
        item.className = 'timeline-item';
        item.title = 'Show this result again';
        const name = entry.uncertain ? t('results.uncertain') : historyDisplayName(entry);
        item.innerHTML = `
          <img src="${entry.thumb}" alt="">
          <span>${formatHistoryWhen(entry.ts)}</span>
          <strong>${name}</strong>`;
        item.addEventListener('click', () => replayHistoryEntry(entry));
        strip.appendChild(item);
      });
      card.appendChild(strip);
      list.appendChild(card);
    });
  }

  /* ---------- Disease library ---------- */
  function renderLibrary() {
    const labels = LeafModel.getLabels();
    const plants = [...new Set(labels.map((l) => l.split('___')[0].replace(/_/g, ' ').replace(/\s*\(.*\)/, '')))].sort();

    const filters = $('library-filters');
    filters.innerHTML = '';
    ['All', ...plants].forEach((plant, idx) => {
      const btn = document.createElement('button');
      btn.className = `filter-chip${idx === 0 ? ' active' : ''}`;
      btn.textContent = plant;
      btn.addEventListener('click', () => {
        filters.querySelectorAll('.filter-chip').forEach((c) => c.classList.toggle('active', c === btn));
        document.querySelectorAll('.library-card').forEach((card) => {
          card.hidden = plant !== 'All' && card.dataset.plant !== plant;
        });
        $('matches-label').textContent = `${t('library.matches')} “${plant}”`;
      });
      filters.appendChild(btn);
    });

    const grid = $('library-grid');
    grid.innerHTML = '';
    [...labels].sort().forEach((label) => {
      const info = LeafModel.getTreatment(label);
      const photo = LIBRARY_IMAGES[label];
      if (!info || !photo) return;
      // Filter chips are derived from the label, not the translated name, so
      // filtering keeps working in every language.
      const plant = label.split('___')[0].replace(/_/g, ' ').replace(/\s*\(.*\)/, '');
      const card = document.createElement('button');
      card.className = 'library-card';
      card.dataset.plant = plant;
      card.innerHTML = `
        <img class="card-img" src="${photo}" alt="${info.common_name} leaf" loading="lazy">
        <div class="card-body">
          <h3>${info.common_name}</h3>
          <p>${info.description}</p>
        </div>`;
      card.addEventListener('click', () => openDiseaseModal(label));
      grid.appendChild(card);
    });
  }

  /* ---------- Disease modal ---------- */
  const modal = $('disease-modal');
  let modalReturnFocus = null;
  function openDiseaseModal(label) {
    const info = LeafModel.getTreatment(label);
    if (!info) return;
    const healthy = /healthy/i.test(label);
    const photo = LIBRARY_IMAGES[label];
    $('modal-body').innerHTML = `
      ${photo ? `<img class="modal-img" src="${photo}" alt="${info.common_name} leaf">` : ''}
      <span class="diagnosis-badge ${healthy ? 'healthy' : `severity-${info.severity || 'medium'}`}">${healthy ? 'Healthy' : (info.severity || 'medium') + ' severity'}</span>
      <h2 id="modal-title">${info.common_name}</h2>
      ${healthy ? `<p>${info.description}</p>` : treatmentHTML(info)}`;
    modalReturnFocus = document.activeElement;
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
    $('modal-close').focus();
  }
  function closeDiseaseModal() {
    modal.hidden = true;
    document.body.style.overflow = '';
    if (modalReturnFocus && modalReturnFocus.isConnected) modalReturnFocus.focus();
    modalReturnFocus = null;
  }
  $('modal-close').addEventListener('click', closeDiseaseModal);
  modal.addEventListener('click', (e) => { if (e.target === modal) closeDiseaseModal(); });
  document.addEventListener('keydown', (e) => {
    if (modal.hidden) return;
    if (e.key === 'Escape') { closeDiseaseModal(); return; }
    // Trap Tab focus inside the dialog while it is open.
    if (e.key === 'Tab') {
      const focusables = modal.querySelectorAll('button, a[href], [tabindex]:not([tabindex="-1"])');
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  /* ---------- Service worker ---------- */
  // Registered on https and localhost. Besides offline caching, the worker
  // injects COOP/COEP headers so repeat visits run multi-threaded WASM.
  if ('serviceWorker' in navigator &&
      (location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname))) {
    navigator.serviceWorker.register('sw.js').catch(() => { /* non-fatal */ });
  }

  I18n.init();
  buildLanguageSwitcher();
  renderHero();
  renderHistory();
  boot();
})();
