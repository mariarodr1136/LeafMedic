/*
 * LeafMedic — image quality & out-of-distribution heuristics.
 *
 * Direct counterpart of image_quality.py; the thresholds below are mirrored
 * there and tests/test_quality.py asserts the two files stay in sync.
 *
 * Three independent questions:
 *   Is this a leaf at all?      -> vegetation coverage
 *   Is the model actually sure? -> normalized predictive entropy
 *   Is the photo usable?        -> blur (Laplacian variance) and exposure
 */
'use strict';

const LeafQuality = (() => {
  // --- Thresholds (keep in sync with image_quality.py) ---
  const LEAF_SCORE_MIN = 0.12;
  const ENTROPY_MAX = 0.75;
  const BLUR_VAR_MIN = 10.0;
  const DARK_MEAN_MAX = 45.0;
  const BRIGHT_MEAN_MIN = 215.0;
  const CLIPPED_FRAC_MAX = 0.35;

  // Vegetation hue window on OpenCV's 0-179 scale, so the Python and JS
  // implementations can share one set of numbers.
  const VEG_HUE_MIN = 15;
  const VEG_HUE_MAX = 100;
  const VEG_SAT_MIN = 45;
  const VEG_VAL_MIN = 40;

  /* RGB (0-255) -> HSV on OpenCV's scale: H in 0-179, S and V in 0-255. */
  function rgbToHsv(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = 60 * (((g - b) / d) % 6);
      else if (max === g) h = 60 * ((b - r) / d + 2);
      else h = 60 * ((r - g) / d + 4);
      if (h < 0) h += 360;
    }
    return { h: h / 2, s: max === 0 ? 0 : (d / max) * 255, v: max };
  }

  /* Per-pixel scan of an RGBA buffer producing every scalar metric at once.
   * Runs in the same pass as tensor construction, so it costs one traversal
   * and no extra inference. */
  function scanPixels(data, width, height) {
    const n = width * height;
    const luma = new Float32Array(n);
    let vegetation = 0;
    let clipped = 0;
    let lumaSum = 0;

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const { h, s, v } = rgbToHsv(r, g, b);
      if (h >= VEG_HUE_MIN && h <= VEG_HUE_MAX && s >= VEG_SAT_MIN && v >= VEG_VAL_MIN) {
        vegetation++;
      }
      // Rec. 601 luma, matching cv2.cvtColor(..., COLOR_RGB2GRAY).
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      luma[p] = y;
      lumaSum += y;
      if (y <= 2 || y >= 253) clipped++;
    }

    return {
      leafScore: vegetation / n,
      meanLuma: lumaSum / n,
      clippedFraction: clipped / n,
      blurScore: laplacianVariance(luma, width, height),
    };
  }

  /* Variance of the 4-neighbour Laplacian — the standard focus measure.
   * Sharp edges give a wide second-derivative distribution, a defocused
   * image a narrow one. Border pixels are skipped, as OpenCV's default
   * BORDER_REFLECT contributes negligibly at 300x300. */
  function laplacianVariance(luma, width, height) {
    let sum = 0;
    let sumSq = 0;
    let count = 0;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        const lap =
          luma[i - width] + luma[i + width] + luma[i - 1] + luma[i + 1] - 4 * luma[i];
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    if (!count) return 0;
    const mean = sum / count;
    return sumSq / count - mean * mean;
  }

  /* Shannon entropy of a probability vector, normalized to [0, 1].
   * ~0 = one confident class, ~1 = uniform (the model has no idea). */
  function normalizedEntropy(probs) {
    const sum = probs.reduce((a, p) => a + p, 0);
    if (sum <= 0) return 1;
    let h = 0;
    for (const p of probs) {
      const q = p / sum;
      if (q > 0) h -= q * Math.log(q);
    }
    return h / Math.log(probs.length);
  }

  /* Combine the pixel metrics and the prediction into flags plus
   * human-readable warnings. Mirrors image_quality.assess(). */
  function assess(metrics, { topConfidence = 1, entropy = 0, confidenceFloor = 0.3 } = {}) {
    const notLeaf = metrics.leafScore < LEAF_SCORE_MIN;
    const blurry = metrics.blurScore < BLUR_VAR_MIN;
    const tooDark = metrics.meanLuma < DARK_MEAN_MAX;
    const tooBright =
      metrics.meanLuma > BRIGHT_MEAN_MIN || metrics.clippedFraction > CLIPPED_FRAC_MAX;
    const uncertain = entropy > ENTROPY_MAX || topConfidence < confidenceFloor;

    const warnings = [];
    if (notLeaf) {
      warnings.push(
        'This image doesn’t look like a close-up photo of a leaf, so the diagnosis below is unreliable.'
      );
    }
    if (blurry) warnings.push('The photo looks out of focus — hold the camera steady and retake it.');
    if (tooDark) warnings.push('The photo is underexposed — use brighter, even lighting.');
    if (tooBright) warnings.push('The photo is overexposed — avoid direct glare and harsh sunlight.');

    return {
      ...metrics,
      entropy,
      topConfidence,
      notLeaf,
      blurry,
      tooDark,
      tooBright,
      uncertain,
      trustworthy: !(notLeaf || uncertain),
      captureOk: !(blurry || tooDark || tooBright),
      warnings,
    };
  }

  return {
    scanPixels,
    normalizedEntropy,
    assess,
    thresholds: {
      LEAF_SCORE_MIN, ENTROPY_MAX, BLUR_VAR_MIN, DARK_MEAN_MAX,
      BRIGHT_MEAN_MIN, CLIPPED_FRAC_MAX,
      VEG_HUE_MIN, VEG_HUE_MAX, VEG_SAT_MIN, VEG_VAL_MIN,
    },
  };
})();

// Node (tests) and browser both load this file.
if (typeof module !== 'undefined' && module.exports) module.exports = LeafQuality;
