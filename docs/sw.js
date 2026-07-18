/* LeafMedic service worker — caches the app shell up front and the heavy
 * assets (model, WASM, samples) on first use, so repeat visits work offline. */
'use strict';

const CACHE = 'leafmedic-v2';
const SHELL = [
  '.',
  'index.html',
  'css/style.css',
  'js/app.js',
  'js/inference.js',
  'data/labels.json',
  'data/treatments.json',
  'icons/icon.svg',
  'manifest.json',
];

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
  // Heavy immutable assets (model, WASM, samples): cache-first.
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
          return res;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(request, clone));
        }
        return res;
      });
    })
  );
});
