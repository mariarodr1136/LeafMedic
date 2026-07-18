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

  // Library card photos for conditions we have sample images of.
  const LIBRARY_IMAGES = {
    'Corn_(maize)___Common_rust_': 'samples/corn_rust_1.jpg',
    'Tomato___Bacterial_spot': 'samples/tomato_bacterial_spot_1.jpg',
    'Tomato___Leaf_Mold': 'samples/tomato_leaf_mold_1.jpg',
  };

  const HERO_TEXT = {
    detect: {
      crumb: 'Disease Identifier',
      title: 'Plant Disease Identification Tool',
      copy: "Don't let your crops suffer from disorders and disease damage. Begin treatment with our free plant disease identifier — the AI runs right in your browser, and your photos never leave your device.",
    },
    library: {
      crumb: 'Disease Library',
      title: 'Plant Disease Library',
      copy: 'Every condition the LeafMedic model can recognize, with symptoms, treatment, and prevention guidance for each one.',
    },
    about: {
      crumb: 'About',
      title: 'About LeafMedic',
      copy: 'An on-device AI plant disease identifier that began life on a Raspberry Pi — now running entirely in your browser.',
    },
  };

  /* ---------- View tabs ---------- */
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  });
  $('brand-link').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  $('crumb-home').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); });
  $('footer-brand-link').addEventListener('click', (e) => { e.preventDefault(); showView('detect'); window.scrollTo({ top: 0, behavior: 'smooth' }); });
  document.querySelectorAll('.footer-col a[data-goto]').forEach((a) => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      showView(a.dataset.goto);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (a.classList.contains('js-scroll-tool')) {
        setTimeout(() => $('detect-tool').scrollIntoView({ behavior: 'smooth' }), 50);
      }
    });
  });
  function showView(name) {
    document.querySelectorAll('.tab').forEach((t) => {
      const active = t.dataset.view === name;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
    const hero = HERO_TEXT[name] || HERO_TEXT.detect;
    $('crumb-current').textContent = hero.crumb;
    $('hero-title').textContent = hero.title;
    $('hero-copy').textContent = hero.copy;
    $('hero-cta').hidden = name === 'about';
    if (name !== 'detect') stopCamera();
  }

  /* ---------- Hero CTA ---------- */
  $('hero-cta').addEventListener('click', () => {
    showView('detect');
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
      const { preds, leafScore, entropy } = await LeafModel.classify(source, 3);
      renderResults(preds, thumb, { leafScore, entropy });
      pushHistory(preds, makeThumbnail(source, 96));
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

  function renderResults(preds, thumbUrl, meta) {
    const top = preds[0];
    const info = LeafModel.getTreatment(top.label);
    const healthy = /healthy/i.test(top.label);
    // Out-of-distribution guard: top confidence too low, prediction spread
    // too flat, or the image barely contains vegetation-colored pixels.
    const notLeaf = meta && meta.leafScore < 0.12;
    const lowConfidence = top.confidence < CONFIDENCE_FLOOR ||
      notLeaf || (meta && meta.entropy > 0.75);

    $('results-empty').hidden = true;
    $('results-content').hidden = false;
    $('analyzed-img').src = thumbUrl;

    const note = $('low-confidence-note');
    note.innerHTML = notLeaf
      ? 'This image doesn’t look like a close-up photo of a leaf, so the diagnosis below is unreliable. Photograph a single leaf filling most of the frame — the model only knows <strong>tomato, corn, soybean, and cabbage</strong> leaves.'
      : 'The model isn’t confident about this image. Make sure the photo shows a single leaf, well lit and in focus — and note the model only knows <strong>tomato, corn, soybean, and cabbage</strong> leaves.';

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
  function pushHistory(preds, thumbUrl) {
    const top = preds[0];
    const entries = getHistory();
    entries.unshift({
      name: top.name, label: top.label, confidence: top.confidence,
      preds: preds.map(({ name, label, confidence }) => ({ name, label, confidence })),
      thumb: thumbUrl, ts: Date.now(),
    });
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
    entries.forEach((entry) => {
      const { name, confidence, thumb, ts } = entry;
      const item = document.createElement('button');
      item.className = 'history-item';
      item.title = 'Show this result again';
      item.innerHTML = `
        <img src="${thumb}" alt="">
        <div class="history-meta">
          <strong>${name}</strong>
          <span>${(confidence * 100).toFixed(0)}% · ${new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span>
        </div>`;
      item.addEventListener('click', () => {
        // Older entries stored only the top prediction — rebuild a preds list.
        const preds = entry.preds || [{ name: entry.name, label: entry.label, confidence: entry.confidence }];
        renderResults(preds, thumb);
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      });
      grid.appendChild(item);
    });
  }
  $('history-clear').addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  /* ---------- Disease library ---------- */
  const LEAF_PLACEHOLDER = `
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C7 6 4 10.5 4 15a8 8 0 0 0 16 0c0-4.5-3-9-8-13Z"/><path d="M12 7v11" stroke-linecap="round"/></svg>`;

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
        $('matches-label').textContent = `diseases matches “${plant}”`;
      });
      filters.appendChild(btn);
    });

    const grid = $('library-grid');
    grid.innerHTML = '';
    [...labels].sort().forEach((label) => {
      const info = LeafModel.getTreatment(label);
      if (!info) return;
      const plant = label.split('___')[0].replace(/_/g, ' ').replace(/\s*\(.*\)/, '');
      const photo = LIBRARY_IMAGES[label];
      const card = document.createElement('button');
      card.className = 'library-card';
      card.dataset.plant = plant;
      card.innerHTML = `
        ${photo
          ? `<img class="card-img" src="${photo}" alt="${info.common_name} leaf" loading="lazy">`
          : `<div class="card-img placeholder">${LEAF_PLACEHOLDER}</div>`}
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

  renderHistory();
  boot();
})();
