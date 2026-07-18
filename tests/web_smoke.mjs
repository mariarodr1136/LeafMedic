/*
 * Browser-demo smoke test: serves docs/ locally, loads the page in headless
 * Chromium, and exercises the full user path — diagnosis, out-of-distribution
 * guard, explainability heatmap, language switch, and the disease library.
 *
 * Usage: node tests/web_smoke.mjs   (requires `playwright` to be installed)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 3456;

const server = spawn('python3', ['-m', 'http.server', String(PORT), '-d', path.join(repo, 'docs')], {
  stdio: 'ignore',
});

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  server.kill();
  process.exit(1);
}

// ONNX Runtime logs benign build/EP notices on stderr; they are not page errors.
const IGNORED_CONSOLE = [
  /VerifyEachNodeIsAssignedToAnEp/,
  /Rerunning with verbose output/,
  /Some nodes were not assigned/,
];
const isRealError = (text) => !IGNORED_CONSOLE.some((re) => re.test(text));

try {
  // Wait for the server to accept connections
  for (let i = 0; ; i++) {
    try {
      const res = await fetch(`http://localhost:${PORT}/`);
      if (res.ok) break;
    } catch {
      if (i > 30) fail('dev server did not start');
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error' && isRealError(m.text())) errors.push(m.text());
  });

  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForSelector('body.model-ready', { timeout: 60000 })
    .catch(() => fail('model never became ready'));

  const backend = await page.evaluate(() => LeafModel.getBackend());
  if (!['wasm', 'webgpu'].includes(backend)) fail(`unexpected backend "${backend}"`);

  // --- A known sample produces its known diagnosis -------------------------
  await page.click('.sample-item >> nth=0');
  await page.waitForFunction(() => !document.getElementById('results-content').hidden,
    { timeout: 60000 }).catch(() => fail('results never rendered'));
  const diagnosis = await page.innerText('#diagnosis-name').catch(() => '');
  if (!/corn common rust/i.test(diagnosis)) fail(`expected Corn Common Rust, got "${diagnosis}"`);

  const badge = await page.innerText('#diagnosis-badge');
  if (/uncertain/i.test(badge)) fail('a known-good sample was flagged Uncertain');

  // --- Explainability: the heatmap renders over the analyzed leaf ----------
  if (await page.locator('#explain-block').isHidden()) fail('explain block hidden for a confident result');
  await page.click('#explain-btn');
  await page.waitForFunction(() => !document.getElementById('explain-result').hidden,
    { timeout: 120000 }).catch(() => fail('occlusion heatmap never rendered'));
  const heat = await page.evaluate(() => {
    const c = document.getElementById('explain-canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let nonZero = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 0) nonZero++;
    return { w: c.width, h: c.height, nonZero };
  });
  if (heat.w !== 300 || heat.h !== 300) fail(`heatmap canvas is ${heat.w}x${heat.h}, expected 300x300`);
  if (heat.nonZero < 1000) fail('heatmap canvas looks blank');

  // --- Out-of-distribution guard: a non-leaf image must not be diagnosed ---
  // A flat blue field stands in for sky: no vegetation-coloured pixels at all.
  await page.evaluate(async () => {
    const c = document.createElement('canvas');
    c.width = 300; c.height = 300;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#4a8fe0';
    ctx.fillRect(0, 0, 300, 300);
    const img = new Image();
    await new Promise((r) => { img.onload = r; img.src = c.toDataURL(); });
    window.__oodResult = await LeafModel.classify(img, 3);
  });
  const ood = await page.evaluate(() => ({
    leafScore: window.__oodResult.quality.leafScore,
    notLeaf: window.__oodResult.quality.notLeaf,
    trustworthy: window.__oodResult.quality.trustworthy,
  }));
  if (!ood.notLeaf) fail(`sky image scored ${ood.leafScore.toFixed(3)} vegetation — OOD guard missed it`);
  if (ood.trustworthy) fail('a non-leaf image was reported as trustworthy');

  // --- Language switch translates UI and care guidance ---------------------
  await page.selectOption('#lang-select', 'es');
  await page.waitForFunction(
    () => document.querySelector('.tab').textContent.includes('Identificador'),
    { timeout: 10000 }
  ).catch(() => fail('UI did not switch to Spanish'));
  // Interface strings are swapped synchronously, but disease names come from
  // treatments.es.json and repaint only once that fetch resolves. Wait for the
  // diagnosis itself rather than assuming the two land together.
  await page.waitForFunction(
    () => /roya común/i.test(document.getElementById('diagnosis-name').textContent),
    { timeout: 10000 }
  ).catch(async () => fail(
    `expected Spanish diagnosis, got "${await page.innerText('#diagnosis-name')}" — ` +
    'the knowledge base did not repaint after the language switch'
  ));
  const esDiagnosis = await page.innerText('#diagnosis-name');
  await page.selectOption('#lang-select', 'en');
  await page.waitForFunction(
    () => document.getElementById('diagnosis-name').textContent.includes('Corn'),
    { timeout: 10000 }
  ).catch(() => fail('UI did not switch back to English'));

  // --- Library renders cards and the modal opens ---------------------------
  await page.click('button.tab:has-text("Disease Library")');
  const cards = await page.locator('.library-card').count();
  if (cards < 16) fail(`expected 16 library cards, got ${cards}`);
  await page.click('.library-card >> nth=0');
  if (await page.locator('#disease-modal').isHidden()) fail('modal did not open');
  await page.keyboard.press('Escape');

  if (errors.length) fail(`console errors:\n${errors.join('\n')}`);

  console.log(
    `PASS: backend=${backend}, diagnosis="${diagnosis}", heatmap ok, ` +
    `OOD guard caught non-leaf (veg=${ood.leafScore.toFixed(3)}), ` +
    `es/en switch ok, ${cards} library cards, no console errors`
  );
  await browser.close();
  server.kill();
  process.exit(0);
} catch (err) {
  fail(err.stack || String(err));
}
