/*
 * LeafMedic — lightweight internationalization.
 *
 * A plain lookup table plus a `data-i18n` attribute sweep over the DOM. No
 * framework and no build step: strings live here, elements name their key,
 * and applyTranslations() rewrites them whenever the language changes.
 *
 * Disease names and care guidance are NOT here — those live in
 * data/treatments.<lang>.json alongside the English knowledge base, because
 * they are data rather than interface chrome.
 */
'use strict';

const I18n = (() => {
  const STORAGE_KEY = 'leafmedic-lang';
  const DEFAULT_LANG = 'en';

  const STRINGS = {
    en: {
      'nav.detect': 'Disease Identifier',
      'nav.diagnose': 'Diagnose Plant',
      'nav.library': 'Disease Library',
      'nav.about': 'About',
      'promo': 'Identify, Treat & Grow Healthy Crops with LeafMedic!',
      'hero.detect.title': 'Plant Disease Identification Tool',
      'hero.detect.copy': "Don't let your crops suffer from disorders and disease damage. Begin treatment with our free plant disease identifier — the AI runs right in your browser, and your photos never leave your device.",
      'hero.diagnose.title': 'Diagnose a Plant',
      'hero.diagnose.copy': 'Upload a photo, use your camera, or try a sample — the diagnosis runs in seconds, right in your browser, and your photos never leave your device.',
      'hero.library.title': 'Plant Disease Library',
      'hero.library.copy': 'Every condition the LeafMedic model can recognize, with symptoms, treatment, and prevention guidance for each one.',
      'hero.about.title': 'About LeafMedic',
      'hero.about.copy': 'An on-device AI plant disease identifier that began life on a Raspberry Pi — now running entirely in your browser.',
      'hero.cta': 'Diagnose a plant',
      'model.loading': 'Loading AI model…',
      'model.loadingDetail': 'Downloading once (~12 MB) — cached for future visits',
      'model.failed': 'Failed to load the model',
      'input.samples': 'Samples',
      'input.upload': 'Upload',
      'input.camera': 'Camera',
      'samples.hint': 'No leaf handy? Try one of these real test photos:',
      'upload.drop': 'Drop a leaf photo here',
      'upload.browse': 'or click to browse · you can also paste from clipboard',
      'upload.multi': 'Tip: select 2–5 photos of the same plant for a sturdier combined diagnosis',
      'camera.off': 'Camera is off',
      'camera.enable': 'Enable camera',
      'camera.flip': 'Flip',
      'camera.capture': 'Capture & Analyze',
      'results.emptyTitle': 'Pick an image to begin',
      'results.emptyCopy': 'Choose a sample, upload a photo, or use your camera. The diagnosis runs entirely on your device — your photos never leave your browser.',
      'results.confidence': 'Confidence',
      'results.explain': 'Why this diagnosis?',
      'results.explainHint': 'Highlights the leaf regions the model relied on.',
      'results.explaining': 'Analyzing regions…',
      'results.legendLow': 'less important',
      'results.legendHigh': 'more important',
      'results.explainCaption': 'Brighter areas changed the prediction most when hidden from the model.',
      'results.uncertain': 'Uncertain',
      'results.noDiagnosis': 'No reliable diagnosis',
      'results.healthy': 'Healthy',
      'results.severitySuffix': 'severity',
      'results.confidenceSuffix': 'confidence',
      'results.healthyTitle': 'This leaf looks healthy!',
      'results.healthyCopy': 'Keep up regular watering, good airflow, and periodic checks of leaf undersides to catch problems early.',
      'results.notLeaf': 'This image doesn’t look like a close-up photo of a leaf, so no diagnosis is shown. Photograph a single leaf filling most of the frame — the model only knows <strong>tomato, corn, soybean, and cabbage</strong> leaves.',
      'results.lowConfidence': 'The model isn’t confident about this image, so no diagnosis is shown. Make sure the photo shows a single leaf, well lit and in focus — and note the model only knows <strong>tomato, corn, soybean, and cabbage</strong> leaves.',
      'results.disclaimer': '⚠️ Educational demo only — not a substitute for professional agronomic advice.',
      'results.multiNote': 'Combined diagnosis from {n} photos of this plant.',
      'results.multiSkipped': '{m} didn’t look like a leaf and were left out.',
      'results.report': 'Save report (PDF)',
      'results.reportHint': 'Opens your browser’s print dialog — choose “Save as PDF” to keep or share this diagnosis.',
      'report.title': 'Diagnosis report',
      'report.generated': 'Generated',
      'quality.blurry': 'The photo looks out of focus — hold the camera steady and retake it.',
      'quality.dark': 'The photo is underexposed — use brighter, even lighting.',
      'quality.bright': 'The photo is overexposed — avoid direct glare and harsh sunlight.',
      'treatment.about': 'About',
      'treatment.symptoms': 'Symptoms',
      'treatment.treatment': 'Treatment',
      'treatment.prevention': 'Prevention',
      'history.title': 'Recent analyses',
      'history.clear': 'Clear',
      'history.tagPlaceholder': 'Name this plant…',
      'timelines.title': 'Plant timelines',
      'timelines.count': '{n} diagnoses',
      'library.matches': 'diseases matches',
      'lang.label': 'Language',
    },
    es: {
      'nav.detect': 'Identificador de enfermedades',
      'nav.diagnose': 'Diagnosticar planta',
      'nav.library': 'Biblioteca de enfermedades',
      'nav.about': 'Acerca de',
      'promo': '¡Identifica, trata y cultiva plantas sanas con LeafMedic!',
      'hero.detect.title': 'Herramienta de identificación de enfermedades',
      'hero.detect.copy': 'No dejes que tus cultivos sufran daños por enfermedades. Empieza el tratamiento con nuestro identificador gratuito — la IA se ejecuta en tu navegador y tus fotos nunca salen de tu dispositivo.',
      'hero.diagnose.title': 'Diagnosticar una planta',
      'hero.diagnose.copy': 'Sube una foto, usa tu cámara o prueba una muestra — el diagnóstico se ejecuta en segundos, directamente en tu navegador, y tus fotos nunca salen de tu dispositivo.',
      'hero.library.title': 'Biblioteca de enfermedades',
      'hero.library.copy': 'Todas las condiciones que el modelo de LeafMedic reconoce, con síntomas, tratamiento y prevención para cada una.',
      'hero.about.title': 'Acerca de LeafMedic',
      'hero.about.copy': 'Un identificador de enfermedades con IA en el dispositivo que nació en una Raspberry Pi — ahora funciona íntegramente en tu navegador.',
      'hero.cta': 'Diagnosticar una planta',
      'model.loading': 'Cargando modelo de IA…',
      'model.loadingDetail': 'Se descarga una vez (~12 MB) — queda en caché para las próximas visitas',
      'model.failed': 'No se pudo cargar el modelo',
      'input.samples': 'Ejemplos',
      'input.upload': 'Subir',
      'input.camera': 'Cámara',
      'samples.hint': '¿No tienes una hoja a mano? Prueba con estas fotos reales:',
      'upload.drop': 'Arrastra aquí una foto de una hoja',
      'upload.browse': 'o haz clic para elegir · también puedes pegar desde el portapapeles',
      'upload.multi': 'Consejo: elige de 2 a 5 fotos de la misma planta para un diagnóstico combinado más sólido',
      'camera.off': 'La cámara está apagada',
      'camera.enable': 'Activar cámara',
      'camera.flip': 'Cambiar',
      'camera.capture': 'Capturar y analizar',
      'results.emptyTitle': 'Elige una imagen para empezar',
      'results.emptyCopy': 'Elige un ejemplo, sube una foto o usa tu cámara. El diagnóstico se hace por completo en tu dispositivo — tus fotos nunca salen del navegador.',
      'results.confidence': 'Confianza',
      'results.explain': '¿Por qué este diagnóstico?',
      'results.explainHint': 'Resalta las zonas de la hoja en las que se basó el modelo.',
      'results.explaining': 'Analizando regiones…',
      'results.legendLow': 'menos importante',
      'results.legendHigh': 'más importante',
      'results.explainCaption': 'Las zonas más brillantes son las que más cambiaron la predicción al ocultarse del modelo.',
      'results.uncertain': 'Incierto',
      'results.noDiagnosis': 'Sin diagnóstico fiable',
      'results.healthy': 'Sana',
      'results.severitySuffix': 'de gravedad',
      'results.confidenceSuffix': 'de confianza',
      'results.healthyTitle': '¡Esta hoja se ve sana!',
      'results.healthyCopy': 'Mantén un riego regular, buena ventilación y revisa periódicamente el envés de las hojas para detectar problemas a tiempo.',
      'results.notLeaf': 'Esta imagen no parece una foto de cerca de una hoja, así que no se muestra un diagnóstico. Fotografía una sola hoja que llene el encuadre — el modelo solo conoce hojas de <strong>tomate, maíz, soja y col</strong>.',
      'results.lowConfidence': 'El modelo no está seguro de esta imagen, así que no se muestra un diagnóstico. Asegúrate de que la foto muestre una sola hoja, bien iluminada y enfocada — y ten en cuenta que el modelo solo conoce hojas de <strong>tomate, maíz, soja y col</strong>.',
      'results.disclaimer': '⚠️ Demostración educativa — no sustituye el asesoramiento agronómico profesional.',
      'results.multiNote': 'Diagnóstico combinado de {n} fotos de esta planta.',
      'results.multiSkipped': '{m} no parecían una hoja y se descartaron.',
      'results.report': 'Guardar informe (PDF)',
      'results.reportHint': 'Abre el diálogo de impresión del navegador — elige «Guardar como PDF» para conservar o compartir este diagnóstico.',
      'report.title': 'Informe de diagnóstico',
      'report.generated': 'Generado el',
      'quality.blurry': 'La foto está desenfocada — sujeta la cámara con firmeza y repite la toma.',
      'quality.dark': 'La foto está subexpuesta — usa una iluminación más brillante y uniforme.',
      'quality.bright': 'La foto está sobreexpuesta — evita los reflejos y la luz solar directa.',
      'treatment.about': 'Acerca de',
      'treatment.symptoms': 'Síntomas',
      'treatment.treatment': 'Tratamiento',
      'treatment.prevention': 'Prevención',
      'history.title': 'Análisis recientes',
      'history.clear': 'Borrar',
      'history.tagPlaceholder': 'Ponle un nombre a esta planta…',
      'timelines.title': 'Cronología por planta',
      'timelines.count': '{n} diagnósticos',
      'library.matches': 'enfermedades coinciden con',
      'lang.label': 'Idioma',
    },
  };

  let current = DEFAULT_LANG;
  const listeners = [];

  function supported() { return Object.keys(STRINGS); }

  /* Stored choice wins; otherwise fall back to the browser's language, and
   * finally to English. */
  function detect() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && STRINGS[stored]) return stored;
    } catch { /* private mode — fall through to browser language */ }
    const nav = (navigator.language || DEFAULT_LANG).slice(0, 2).toLowerCase();
    return STRINGS[nav] ? nav : DEFAULT_LANG;
  }

  function t(key) {
    const table = STRINGS[current] || STRINGS[DEFAULT_LANG];
    return table[key] ?? STRINGS[DEFAULT_LANG][key] ?? key;
  }

  function getLang() { return current; }

  function setLang(lang) {
    if (!STRINGS[lang]) return;
    current = lang;
    try { localStorage.setItem(STORAGE_KEY, lang); } catch { /* non-fatal */ }
    document.documentElement.lang = lang;
    applyTranslations();
    listeners.forEach((fn) => fn(lang));
  }

  function onChange(fn) { listeners.push(fn); }

  /* Rewrite every element carrying a data-i18n key. `data-i18n-html` opts into
   * innerHTML for the few strings with inline <strong> emphasis. */
  function applyTranslations(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((el) => {
      el.textContent = t(el.dataset.i18n);
    });
    root.querySelectorAll('[data-i18n-html]').forEach((el) => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    root.querySelectorAll('[data-i18n-aria]').forEach((el) => {
      el.setAttribute('aria-label', t(el.dataset.i18nAria));
    });
  }

  function init() {
    current = detect();
    document.documentElement.lang = current;
    applyTranslations();
  }

  return { init, t, setLang, getLang, supported, onChange, applyTranslations };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = I18n;
