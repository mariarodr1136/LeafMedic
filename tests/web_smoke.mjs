/*
 * Browser-demo smoke test: serves docs/ locally, loads the page in headless
 * Chromium, runs a sample diagnosis end-to-end, and checks the result.
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
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForSelector('body.model-ready', { timeout: 60000 }).catch(() => fail('model never became ready'));

  // Run a sample diagnosis
  await page.click('.sample-item >> nth=0');
  await page.waitForTimeout(5000);
  const diagnosis = await page.innerText('#diagnosis-name').catch(() => '');
  if (!/corn common rust/i.test(diagnosis)) fail(`expected Corn Common Rust, got "${diagnosis}"`);

  // Library renders cards and the modal opens
  await page.click('button.tab:has-text("Disease Library")');
  const cards = await page.locator('.library-card').count();
  if (cards < 16) fail(`expected 16 library cards, got ${cards}`);
  await page.click('.library-card >> nth=0');
  if (await page.locator('#disease-modal').isHidden()) fail('modal did not open');
  await page.keyboard.press('Escape');

  if (errors.length) fail(`console errors:\n${errors.join('\n')}`);

  console.log(`PASS: diagnosis="${diagnosis}", ${cards} library cards, no console errors`);
  await browser.close();
  server.kill();
  process.exit(0);
} catch (err) {
  fail(err.stack || String(err));
}
