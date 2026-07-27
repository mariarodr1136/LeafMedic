/* LeafMedic service worker — caches the app shell up front and the heavy
 * assets (model, WASM, samples) on first use, so repeat visits work offline.
 *
 * It also injects COOP/COEP headers on same-origin responses so the page
 * becomes cross-origin isolated, unlocking multi-threaded WASM inference
 * (GitHub Pages cannot send these headers itself). All assets are
 * same-origin, so require-corp is safe here. */
'use strict';

const CACHE = 'leafmedic-v5';
const SHELL = [
  '.',
  'index.html',
  'css/style.css',
  'js/app.js',
  'js/inference.js',
  'js/quality.js',
  'js/i18n.js',
  'data/labels.json',
  'data/treatments.json',
  'data/treatments.es.json',
  'img/hero-garden.jpg',
  'fonts/nunito-latin.woff2',
  'fonts/outfit-latin.woff2',
  'icons/icon.svg',
  'manifest.json',
];

const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function withIsolationHeaders(response) {
  // Opaque/error responses can't be rebuilt — pass them through.
  if (!response || response.status === 0 || response.type === 'opaque') return response;
  const headers = new Headers(response.headers);
  for (const [k, v] of Object.entries(ISOLATION_HEADERS)) headers.set(k, v);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || !request.url.startsWith(self.location.origin)) return;

  // App shell (html/js/css/data): network-first so site updates reach
  // returning visitors; cached copy is the offline fallback.
  // Heavy immutable assets (model, WASM, samples, fonts): cache-first.
  const path = new URL(request.url).pathname;
  const isShell = request.mode === 'navigate' ||
    /\.(html|js|css|json)$/.test(path);

  if (isShell) {
    event.respondWith(
      fetch(request)
        .then((res) => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(CACHE).then((c) => c.put(request, clone));
          }
          return withIsolationHeaders(res);
        })
        .catch(() => caches.match(request).then(withIsolationHeaders))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return withIsolationHeaders(cached);
      return fetch(request).then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(request, clone));
        }
        return withIsolationHeaders(res);
      });
    })
  );
});
