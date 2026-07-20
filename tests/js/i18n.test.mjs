// Unit tests for docs/js/i18n.js — string lookup and fallback behavior.
// i18n.js touches document/localStorage/navigator, so this stubs the minimum
// DOM surface it needs rather than pulling in a full jsdom dependency.
// Run with: node --test tests/js
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

function installDomStubs({ storedLang, navigatorLang = 'en-US' } = {}) {
  const store = new Map(storedLang ? [['leafmedic-lang', storedLang]] : []);
  const define = (name, value) =>
    Object.defineProperty(global, name, { value, writable: true, configurable: true });
  define('localStorage', {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, v),
    removeItem: (k) => store.delete(k),
  });
  // Node's own global `navigator` (added for fetch parity) is a read-only
  // getter, so a plain assignment throws — defineProperty replaces it.
  define('navigator', { language: navigatorLang });
  define('document', {
    documentElement: { lang: '' },
    querySelectorAll: () => [],
  });
}

const require = createRequire(import.meta.url);

beforeEach(() => {
  installDomStubs();
  // Force a fresh module instance per test so `current` language state
  // (an IIFE-closure singleton) doesn't leak between tests.
  delete require.cache[require.resolve('../../docs/js/i18n.js')];
});

test('t(): returns the English string for a known key by default', () => {
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  assert.equal(I18n.t('nav.about'), 'About');
});

test('t(): an unknown key falls back to the key itself, not a crash', () => {
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  assert.equal(I18n.t('nonexistent.key'), 'nonexistent.key');
});

test('setLang(): switches the active language and updates lookups', () => {
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  I18n.setLang('es');
  assert.equal(I18n.getLang(), 'es');
  assert.equal(I18n.t('nav.about'), 'Acerca de');
});

test('setLang(): an unsupported language code is ignored, keeping the current one', () => {
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  I18n.setLang('fr'); // not in STRINGS
  assert.equal(I18n.getLang(), 'en');
});

test('detect(): a stored language preference wins over the browser language', () => {
  installDomStubs({ storedLang: 'es', navigatorLang: 'en-US' });
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  assert.equal(I18n.getLang(), 'es');
});

test('detect(): falls back to the browser language when nothing is stored', () => {
  installDomStubs({ navigatorLang: 'es-MX' });
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  assert.equal(I18n.getLang(), 'es');
});

test('detect(): an unsupported browser language falls back to English', () => {
  installDomStubs({ navigatorLang: 'fr-FR' });
  const I18n = require('../../docs/js/i18n.js');
  I18n.init();
  assert.equal(I18n.getLang(), 'en');
});

test('every English key has a Spanish counterpart', () => {
  // t() falls back to English silently by design, which would hide a missing
  // Spanish key rather than catch it — so this reads the STRINGS tables
  // directly out of the source instead of going through the lookup API.
  const fs = require('node:fs');
  const src = fs.readFileSync(require.resolve('../../docs/js/i18n.js'), 'utf8');
  const keysOf = (lang) => {
    const block = src.match(new RegExp(`${lang}: \\{([\\s\\S]*?)\\n    \\},`))[1];
    return [...block.matchAll(/^\s*'([^']+)':/gm)].map((m) => m[1]);
  };
  const en = keysOf('en');
  const es = keysOf('es');
  assert.ok(en.length > 0);
  assert.deepEqual([...en].sort(), [...es].sort());
});
