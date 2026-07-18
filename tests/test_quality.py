"""Image quality and out-of-distribution guard tests.

Covers three things the classifier alone cannot protect against:
  * confidently diagnosing something that isn't a leaf,
  * silently accepting an unusable (blurry/dark/blown-out) photo,
  * the Python and JavaScript guards drifting apart.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

import image_quality as iq  # noqa: E402  (import after the importorskip guard)

NEGATIVES = REPO / "test_images" / "_negatives"

# Synthetic non-leaf images and the guard that must reject each one.
# "noise" is deliberately absent: uniform RGB noise contains genuinely
# leaf-coloured pixels and stays sharp, so neither the vegetation nor the blur
# heuristic rejects it. See test_uniform_noise_is_a_known_gap below.
EXPECTED_REJECTIONS = {
    "sky.jpg": "not_leaf",
    "skin.jpg": "not_leaf",
    "pavement.jpg": "not_leaf",
    "screenshot.jpg": "not_leaf",
    "dark_leaf.jpg": "too_dark",
    "blurred_leaf.jpg": "blurry",
}


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"unreadable image {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def real_leaf_images():
    images = [
        p for p in sorted((REPO / "test_images").glob("*/*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.parent.name != "_negatives"
    ]
    assert images, "no leaf test images found"
    return images


# --------------------------------------------------------------------------
# Vegetation coverage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", real_leaf_images(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_real_leaves_score_as_vegetation(path):
    """Every genuine leaf photo must clear the vegetation threshold."""
    score = iq.vegetation_score(load_rgb(path))
    assert score >= iq.LEAF_SCORE_MIN, f"{path.name} scored only {score:.3f} vegetation"


def test_neutral_greys_are_not_vegetation():
    """Regression: the original heuristic (g >= b and g >= r - 20) accepted
    every neutral grey, so concrete and UI screenshots scored ~100% leaf."""
    grey = np.full((64, 64, 3), 140, np.uint8)
    assert iq.vegetation_score(grey) == 0.0
    white = np.full((64, 64, 3), 250, np.uint8)
    assert iq.vegetation_score(white) == 0.0


def test_saturated_green_is_vegetation():
    green = np.zeros((64, 64, 3), np.uint8)
    green[:, :] = (60, 150, 70)
    assert iq.vegetation_score(green) > 0.9


def test_chlorotic_and_necrotic_tissue_still_counts():
    """Diseased leaves are yellow and brown, not green — they must still pass."""
    yellow = np.zeros((64, 64, 3), np.uint8)
    yellow[:, :] = (210, 190, 60)
    brown = np.zeros((64, 64, 3), np.uint8)
    brown[:, :] = (140, 90, 40)
    assert iq.vegetation_score(yellow) > 0.9
    assert iq.vegetation_score(brown) > 0.9


# --------------------------------------------------------------------------
# Full assessment over the synthetic negatives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,flag", sorted(EXPECTED_REJECTIONS.items()))
def test_negative_samples_are_rejected(name, flag):
    path = NEGATIVES / name
    if not path.exists():
        pytest.skip(f"{name} missing — run python3 tests/make_negatives.py")
    report = iq.assess(load_rgb(path))
    assert report[flag], f"{name} was not flagged as {flag}: {report['warnings']}"


@pytest.mark.parametrize("path", real_leaf_images(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_real_leaves_are_not_flagged_as_unusable(path):
    """Guards must not fire on the photos the model is expected to handle."""
    report = iq.assess(load_rgb(path))
    tripped = [k for k in ("not_leaf", "blurry", "too_dark", "too_bright") if report[k]]
    assert not tripped, f"{path.name} tripped {tripped}"


def test_uniform_noise_is_a_known_gap():
    """Documents a real limitation rather than pretending it doesn't exist.

    Uniform RGB noise contains plenty of saturated green-hued pixels and has
    very high Laplacian variance, so it passes both heuristics. Rejecting it
    would need a spatial-coherence signal; a Laplacian ceiling cannot separate
    it from a genuinely detailed leaf photo (real leaves here reach ~19,900
    variance against noise's ~48,800).
    """
    path = NEGATIVES / "noise.jpg"
    if not path.exists():
        pytest.skip("noise.jpg missing — run python3 tests/make_negatives.py")
    report = iq.assess(load_rgb(path))
    assert not report["not_leaf"], (
        "uniform noise is now rejected — the guard improved, update this test"
    )


# --------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------

def test_entropy_bounds():
    n = 16
    uniform = np.full(n, 1.0 / n)
    assert iq.normalized_entropy(uniform) == pytest.approx(1.0, abs=1e-9)

    confident = np.zeros(n)
    confident[3] = 1.0
    assert iq.normalized_entropy(confident) == pytest.approx(0.0, abs=1e-9)


def test_entropy_handles_unnormalized_and_empty_input():
    # The model's dequantized scores do not sum to 1, so the metric normalizes.
    assert iq.normalized_entropy(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(1.0)
    assert iq.normalized_entropy(np.zeros(8)) == 1.0


def test_high_entropy_marks_a_prediction_uncertain():
    leaf = np.zeros((64, 64, 3), np.uint8)
    leaf[:, :] = (60, 150, 70)
    flat = np.full(16, 1.0 / 16)
    assert iq.assess(leaf, flat)["uncertain"]

    peaked = np.zeros(16)
    peaked[0] = 0.95
    assert not iq.assess(leaf, peaked)["uncertain"]


def test_low_top_confidence_marks_a_prediction_uncertain():
    leaf = np.zeros((64, 64, 3), np.uint8)
    leaf[:, :] = (60, 150, 70)
    # Peaked enough to keep entropy low, but under the 0.30 confidence floor.
    probs = np.zeros(16)
    probs[0] = 0.2
    probs[1] = 0.02
    assert iq.assess(leaf, probs)["uncertain"]


# --------------------------------------------------------------------------
# Blur and exposure
# --------------------------------------------------------------------------

def test_blur_score_orders_sharp_above_soft():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    soft = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert iq.blur_score(sharp) > iq.blur_score(soft)


def test_exposure_stats_detect_dark_and_blown_out():
    dark = np.full((64, 64, 3), 5, np.uint8)
    bright = np.full((64, 64, 3), 255, np.uint8)
    dark_mean, _ = iq.exposure_stats(dark)
    bright_mean, bright_clipped = iq.exposure_stats(bright)
    assert dark_mean < iq.DARK_MEAN_MAX
    assert bright_mean > iq.BRIGHT_MEAN_MIN
    assert bright_clipped == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Cross-runtime sync
# --------------------------------------------------------------------------

def test_python_and_js_thresholds_match():
    """image_quality.py and docs/js/quality.js must agree numerically.

    A drift here would make the desktop app and the browser demo disagree about
    which photos are trustworthy, which is exactly the mirroring the project
    claims to maintain.
    """
    js = (REPO / "docs" / "js" / "quality.js").read_text()
    for name, value in iq.thresholds().items():
        match = re.search(rf"\b{name}\s*=\s*([0-9.]+)\s*;", js)
        assert match, f"{name} not found in docs/js/quality.js"
        assert float(match.group(1)) == pytest.approx(float(value)), (
            f"{name}: Python has {value}, JavaScript has {match.group(1)}"
        )


def test_web_labels_and_treatments_cover_every_language():
    """Each translated knowledge base must cover exactly the English label set."""
    en = json.loads((REPO / "data" / "treatments.json").read_text())
    for path in sorted((REPO / "docs" / "data").glob("treatments.*.json")):
        translated = json.loads(path.read_text())
        assert set(translated) == set(en), f"{path.name} label set differs from English"
        for label, entry in translated.items():
            assert entry.get("severity") == en[label].get("severity"), (
                f"{path.name}:{label} severity differs from English"
            )
            for field in ("symptoms", "treatments", "prevention"):
                if field in en[label]:
                    assert len(entry.get(field, [])) == len(en[label][field]), (
                        f"{path.name}:{label} has a different number of {field}"
                    )
