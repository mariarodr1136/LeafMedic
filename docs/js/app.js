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

  /* ---------- Theme ---------- */
  const themeToggle = $('theme-toggle');
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('leafmedic-theme', theme);
  }
  const savedTheme = localStorage.getItem('leafmedic-theme');
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  /* ---------- View tabs ---------- */
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  });
  $('brand-link').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  function showView(name) {
    document.querySelectorAll('.tab').forEach((t) => {
      const active = t.dataset.view === name;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
    if (name !== 'detect') stopCamera();
  }

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
    } catch (err) {
      console.error(err);
      $('model-status-title').textContent = 'Failed to load the model';
      $('model-status-detail').textContent = `${err.message} — check your connection and refresh.`;
      banner.classList.add('error');
    }
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
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); fileInput.value = ''; });
  ['dragover', 'dragenter'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragging');
  }));
  dropzone.addEventListener('drop', (e) => {
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('image/'));
    if (file) loadFile(file);
  });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) loadFile(item.getAsFile());
  });
  function loadFile(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { analyze(img); URL.revokeObjectURL(url); };
    img.src = url;
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

  async function analyze(source) {
    if (!inputsEnabled) return;
    resultsPanel.classList.add('analyzing');
    try {
      const thumb = makeThumbnail(source, 320);
      const preds = await LeafModel.classify(source, 3);
      renderResults(preds, thumb);
      pushHistory(preds[0], makeThumbnail(source, 96));
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

  function renderResults(preds, thumbUrl) {
    const top = preds[0];
    const info = LeafModel.getTreatment(top.label);
    const healthy = /healthy/i.test(top.label);
    const lowConfidence = top.confidence < CONFIDENCE_FLOOR;

    $('results-empty').hidden = true;
    $('results-content').hidden = false;
    $('analyzed-img').src = thumbUrl;

    const badge = $('diagnosis-badge');
    if (lowConfidence) {
      badge.textContent = 'Uncertain';
      badge.className = 'diagnosis-badge uncertain';
    } else if (healthy) {
      badge.textContent = 'Healthy';
      badge.className = 'diagnosis-badge healthy';
    } else {
      badge.textContent = (info?.severity || 'disease') + ' severity';
      badge.className = `diagnosis-badge severity-${info?.severity || 'medium'}`;
    }
    $('diagnosis-name').textContent = top.name;
    $('diagnosis-sub').textContent = `${(top.confidence * 100).toFixed(1)}% confidence`;
    $('low-confidence-note').hidden = !lowConfidence;

    const bars = $('pred-bars');
    bars.innerHTML = '';
    preds.forEach(({ name, confidence, label }) => {
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

    $('treatment-card').innerHTML = healthy || !info ? (healthy ? `
      <div class="healthy-note">
        <strong>🌿 This leaf looks healthy!</strong>
        <p>Keep up regular watering, good airflow, and periodic checks of leaf undersides to catch problems early.</p>
      </div>` : '') : treatmentHTML(info);

    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function treatmentHTML(info) {
    const list = (items) => items.map((i) => `<li>${i}</li>`).join('');
    return `
      <h3 class="section-title">About ${info.common_name}</h3>
      <p class="treatment-desc">${info.description}</p>
      <div class="treatment-cols">
        <div class="treatment-col">
          <h4><span aria-hidden="true">🔍</span> Symptoms</h4>
          <ul>${list(info.symptoms || [])}</ul>
        </div>
        <div class="treatment-col">
          <h4><span aria-hidden="true">💊</span> Treatment</h4>
          <ul>${list(info.treatments || [])}</ul>
        </div>
        <div class="treatment-col">
          <h4><span aria-hidden="true">🛡️</span> Prevention</h4>
          <ul>${list(info.prevention || [])}</ul>
        </div>
      </div>`;
  }

  /* ---------- History ---------- */
  function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }
  function pushHistory(top, thumbUrl) {
    const entries = getHistory();
    entries.unshift({ name: top.name, label: top.label, confidence: top.confidence, thumb: thumbUrl, ts: Date.now() });
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, HISTORY_MAX)));
    } catch { /* storage full — drop silently */ }
    renderHistory();
  }
  function renderHistory() {
    const entries = getHistory();
    $('history-block').hidden = entries.length === 0;
    const grid = $('history-grid');
    grid.innerHTML = '';
    entries.forEach(({ name, confidence, thumb, ts }) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `
        <img src="${thumb}" alt="">
        <div class="history-meta">
          <strong>${name}</strong>
          <span>${(confidence * 100).toFixed(0)}% · ${new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span>
        </div>`;
      grid.appendChild(item);
    });
  }
  $('history-clear').addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

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
      });
      filters.appendChild(btn);
    });

    const grid = $('library-grid');
    grid.innerHTML = '';
    [...labels].sort().forEach((label) => {
      const info = LeafModel.getTreatment(label);
      if (!info) return;
      const healthy = /healthy/i.test(label);
      const plant = label.split('___')[0].replace(/_/g, ' ').replace(/\s*\(.*\)/, '');
      const card = document.createElement('details');
      card.className = 'library-card panel';
      card.dataset.plant = plant;
      card.innerHTML = `
        <summary>
          <div>
            <span class="diagnosis-badge ${healthy ? 'healthy' : `severity-${info.severity || 'medium'}`}">${healthy ? 'Healthy' : info.severity + ' severity'}</span>
            <h3>${info.common_name}</h3>
            <p>${info.description}</p>
          </div>
          <svg class="chevron" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </summary>
        <div class="library-card-body">${healthy ? `<p>${info.description}</p>` : treatmentHTML(info)}</div>`;
      grid.appendChild(card);
    });
  }

  /* ---------- Service worker ---------- */
  if ('serviceWorker' in navigator && location.protocol === 'https:') {
    navigator.serviceWorker.register('sw.js').catch(() => { /* non-fatal */ });
  }

  renderHistory();
  boot();
})();
