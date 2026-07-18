#!/usr/bin/env python3

"""
Image Quality & Out-of-Distribution Heuristics
==============================================
Cheap, model-free signals computed alongside preprocessing, used to decide
whether a prediction should be trusted or the user should be asked to reshoot.

These mirror the browser implementation in ``docs/js/quality.js`` — the
thresholds are defined once here and once there, and
``tests/test_quality.py`` asserts the two stay in sync.

Three independent questions:
  * Is this a leaf at all?      -> vegetation coverage
  * Is the model actually sure? -> normalized predictive entropy
  * Is the photo usable?        -> blur (Laplacian variance) and exposure
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# --- Thresholds (keep in sync with docs/js/quality.js) ----------------------
LEAF_SCORE_MIN = 0.12       # below this fraction of vegetation pixels -> "not a leaf"
ENTROPY_MAX = 0.75          # above this normalized entropy -> model is guessing
BLUR_VAR_MIN = 10.0         # Laplacian variance below this -> too blurry
DARK_MEAN_MAX = 45.0        # mean luma below this -> underexposed
BRIGHT_MEAN_MIN = 215.0     # mean luma above this -> overexposed
CLIPPED_FRAC_MAX = 0.35     # fraction of pure-black/white pixels tolerated

# Vegetation hue window, in OpenCV's 0-179 hue scale: from warm
# yellow/orange (necrotic and chlorotic tissue) through to green.
VEG_HUE_MIN = 15
VEG_HUE_MAX = 100
VEG_SAT_MIN = 45            # rejects greys: concrete, screenshots, overcast walls
VEG_VAL_MIN = 40            # rejects near-black pixels


def vegetation_score(rgb: np.ndarray) -> float:
    """Fraction of pixels whose colour looks like plant tissue.

    Works in HSV rather than on raw channel comparisons. An earlier version
    tested ``g >= b and g >= r - 20``, which every neutral grey satisfies —
    concrete and UI screenshots scored ~100% vegetation. Requiring a minimum
    saturation inside a yellow-to-green hue window fixes that while still
    accepting chlorotic (yellow) and necrotic (brown) diseased tissue.

    Args:
        rgb: HxWx3 uint8 image in RGB order.

    Returns:
        Fraction in [0, 1].
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    mask = (h >= VEG_HUE_MIN) & (h <= VEG_HUE_MAX) & (s >= VEG_SAT_MIN) & (v >= VEG_VAL_MIN)
    return float(mask.mean())


def normalized_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector, scaled to [0, 1].

    ~0 means one dominant class; ~1 means the model is guessing uniformly.
    """
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return 1.0
    p = p / total
    nonzero = p[p > 0]
    h = float(-(nonzero * np.log(nonzero)).sum())
    return h / float(np.log(len(p)))


def blur_score(rgb: np.ndarray) -> float:
    """Variance of the Laplacian — the standard focus measure.

    Sharp edges produce a wide second-derivative distribution; a defocused
    image produces a narrow one. Higher is sharper.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(rgb: np.ndarray) -> tuple[float, float]:
    """Mean luma and the fraction of pixels clipped to pure black or white."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    return float(gray.mean()), clipped


def assess(rgb: np.ndarray, probs: np.ndarray | None = None,
           confidence_threshold: float = 0.30) -> dict[str, Any]:
    """Score an image (and optionally its prediction) for trustworthiness.

    Args:
        rgb: HxWx3 uint8 image in RGB order (any size).
        probs: Full probability vector from the model, if available.
        confidence_threshold: Minimum top-1 confidence to trust.

    Returns:
        Dict of raw metrics plus boolean flags and a list of human-readable
        ``warnings``. ``trustworthy`` is False when any guard trips.
    """
    leaf = vegetation_score(rgb)
    sharpness = blur_score(rgb)
    mean_luma, clipped = exposure_stats(rgb)

    entropy = normalized_entropy(probs) if probs is not None else 0.0
    top_confidence = float(np.max(probs)) if probs is not None else 1.0

    not_leaf = leaf < LEAF_SCORE_MIN
    blurry = sharpness < BLUR_VAR_MIN
    too_dark = mean_luma < DARK_MEAN_MAX
    too_bright = mean_luma > BRIGHT_MEAN_MIN or clipped > CLIPPED_FRAC_MAX
    uncertain = entropy > ENTROPY_MAX or top_confidence < confidence_threshold

    warnings: list[str] = []
    if not_leaf:
        warnings.append(
            "This doesn't look like a close-up of a leaf — the diagnosis is unreliable."
        )
    if blurry:
        warnings.append("The photo looks out of focus. Hold the camera steady and retake it.")
    if too_dark:
        warnings.append("The photo is underexposed. Use brighter, even lighting.")
    if too_bright:
        warnings.append("The photo is overexposed. Avoid direct glare and harsh sunlight.")
    if uncertain:
        warnings.append("The model is not confident about this image.")

    return {
        "leaf_score": leaf,
        "entropy": entropy,
        "top_confidence": top_confidence,
        "blur_score": sharpness,
        "mean_luma": mean_luma,
        "clipped_fraction": clipped,
        "not_leaf": not_leaf,
        "blurry": blurry,
        "too_dark": too_dark,
        "too_bright": too_bright,
        "uncertain": uncertain,
        "trustworthy": not (not_leaf or uncertain),
        "capture_ok": not (blurry or too_dark or too_bright),
        "warnings": warnings,
    }


def thresholds() -> dict[str, float]:
    """The threshold table, for cross-runtime sync tests."""
    return {
        "LEAF_SCORE_MIN": LEAF_SCORE_MIN,
        "ENTROPY_MAX": ENTROPY_MAX,
        "BLUR_VAR_MIN": BLUR_VAR_MIN,
        "DARK_MEAN_MAX": DARK_MEAN_MAX,
        "BRIGHT_MEAN_MIN": BRIGHT_MEAN_MIN,
        "CLIPPED_FRAC_MAX": CLIPPED_FRAC_MAX,
        "VEG_HUE_MIN": VEG_HUE_MIN,
        "VEG_HUE_MAX": VEG_HUE_MAX,
        "VEG_SAT_MIN": VEG_SAT_MIN,
        "VEG_VAL_MIN": VEG_VAL_MIN,
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 image_quality.py <image> [image ...]")
        raise SystemExit(1)

    for path in sys.argv[1:]:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{path}: unreadable")
            continue
        report = assess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        print(f"\n{path}")
        print(f"  vegetation : {report['leaf_score']:.3f}  (min {LEAF_SCORE_MIN})")
        print(f"  sharpness  : {report['blur_score']:.1f}   (min {BLUR_VAR_MIN})")
        print(f"  mean luma  : {report['mean_luma']:.1f}   clipped {report['clipped_fraction']:.3f}")
        for w in report["warnings"]:
            print(f"  ! {w}")
        if not report["warnings"]:
            print("  ✓ looks like a usable leaf photo")
