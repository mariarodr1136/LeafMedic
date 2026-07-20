// Unit tests for docs/js/quality.js — previously only exercised indirectly
// by the end-to-end Playwright smoke test. Run with: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const LeafQuality = require('../../docs/js/quality.js');

function makeImage(width, height, [r, g, b]) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
  }
  return data;
}

test('scanPixels: a uniform green leaf-colored patch scores as vegetation', () => {
  const data = makeImage(10, 10, [60, 130, 40]); // leafy green, well inside the HSV window
  const metrics = LeafQuality.scanPixels(data, 10, 10);
  assert.equal(metrics.leafScore, 1);
});

test('scanPixels: uniform gray (no saturation) scores zero vegetation', () => {
  const data = makeImage(10, 10, [128, 128, 128]);
  const metrics = LeafQuality.scanPixels(data, 10, 10);
  assert.equal(metrics.leafScore, 0);
});

test('scanPixels: a flat-color image has zero blur variance (perfectly smooth)', () => {
  const data = makeImage(20, 20, [60, 130, 40]);
  const metrics = LeafQuality.scanPixels(data, 20, 20);
  assert.equal(metrics.blurScore, 0);
});

test('scanPixels: mean luma reflects Rec.601 weighting, not a plain RGB average', () => {
  const data = makeImage(4, 4, [0, 255, 0]); // pure green
  const metrics = LeafQuality.scanPixels(data, 4, 4);
  // luma = 0.299*0 + 0.587*255 + 0.114*0
  assert.ok(Math.abs(metrics.meanLuma - 0.587 * 255) < 1e-6);
});

test('normalizedEntropy: a one-hot distribution has ~zero entropy', () => {
  const h = LeafQuality.normalizedEntropy([1, 0, 0, 0]);
  assert.ok(h < 1e-9);
});

test('normalizedEntropy: a uniform distribution has entropy ~1', () => {
  const h = LeafQuality.normalizedEntropy([0.25, 0.25, 0.25, 0.25]);
  assert.ok(Math.abs(h - 1) < 1e-9);
});

test('normalizedEntropy: an all-zero vector is treated as maximally uncertain', () => {
  assert.equal(LeafQuality.normalizedEntropy([0, 0, 0]), 1);
});

test('assess: a sharp, well-lit, confident leaf photo is trustworthy and capture-ok', () => {
  const result = LeafQuality.assess(
    { leafScore: 0.9, blurScore: 500, meanLuma: 120, clippedFraction: 0 },
    { topConfidence: 0.95, entropy: 0.1 }
  );
  assert.equal(result.notLeaf, false);
  assert.equal(result.blurry, false);
  assert.equal(result.trustworthy, true);
  assert.equal(result.captureOk, true);
  assert.deepEqual(result.warnings, []);
});

test('assess: below the leaf-score threshold is flagged not-a-leaf with a warning', () => {
  const result = LeafQuality.assess(
    { leafScore: 0.01, blurScore: 500, meanLuma: 120, clippedFraction: 0 },
    { topConfidence: 0.95, entropy: 0.1 }
  );
  assert.equal(result.notLeaf, true);
  assert.equal(result.trustworthy, false);
  assert.ok(result.warnings.length === 1);
});

test('assess: high entropy or low confidence marks the result uncertain, independent of image quality', () => {
  const goodImage = { leafScore: 0.9, blurScore: 500, meanLuma: 120, clippedFraction: 0 };
  const highEntropy = LeafQuality.assess(goodImage, { topConfidence: 0.9, entropy: 0.9 });
  assert.equal(highEntropy.uncertain, true);
  assert.equal(highEntropy.trustworthy, false);

  const lowConfidence = LeafQuality.assess(goodImage, { topConfidence: 0.1, entropy: 0.1 });
  assert.equal(lowConfidence.uncertain, true);
});

test('assess: blur/exposure flags do not affect trustworthy, only captureOk', () => {
  const result = LeafQuality.assess(
    { leafScore: 0.9, blurScore: 1, meanLuma: 120, clippedFraction: 0 },
    { topConfidence: 0.95, entropy: 0.1 }
  );
  assert.equal(result.blurry, true);
  assert.equal(result.captureOk, false);
  assert.equal(result.trustworthy, true); // blur alone doesn't imply distrust
});

test('thresholds are exposed for the Python/JS drift test to compare against', () => {
  assert.equal(typeof LeafQuality.thresholds.LEAF_SCORE_MIN, 'number');
  assert.equal(typeof LeafQuality.thresholds.ENTROPY_MAX, 'number');
});
