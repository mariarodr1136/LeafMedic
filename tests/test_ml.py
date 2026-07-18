"""Golden prediction tests: the model must diagnose known sample images correctly.

Requires a TFLite runtime (tensorflow, ai-edge-litert, or tflite-runtime);
skipped automatically when none is installed.

The suite is split in two deliberately:

* GOLDEN — classes the shipped AgriPredict model handles reliably. A
  regression in preprocessing, dequantization, or the model file itself moves
  a whole class at once and fails here.
* KNOWN_WEAK — classes it does *not* handle. These are marked xfail rather
  than deleted, so the gap stays visible and a retrained model that fixes one
  shows up as an unexpected pass (XPASS) instead of silently going unnoticed.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")


def _has_tflite_runtime():
    for mod in ("ai_edge_litert.interpreter", "tensorflow", "tflite_runtime.interpreter"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


pytestmark = pytest.mark.skipif(
    not _has_tflite_runtime(), reason="no TFLite runtime installed"
)

# (test_images subdirectory, expected label)
GOLDEN = [
    ("corn_common_rust", "Corn_(maize)___Common_rust_"),
    ("tomato_bacterial_spot", "Tomato___Bacterial_spot"),
    ("tomato_mold", "Tomato___Leaf_Mold"),
    ("tomato_septoria_leaf_spot", "Tomato___Septoria_leaf_spot"),
    ("tomato_late_blight", "Tomato___Late_blight"),
    ("tomato_spider_mites", "Tomato___Spider_mites"),
    ("tomato_yellow_leaf_curl_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
]

# Classes the shipped model gets wrong on PlantVillage imagery. Measured
# 2026-07-18 over three images each: healthy tomato 1/3, healthy corn 0/3,
# corn gray leaf spot 0/3 (confused with common rust), healthy soybean 0/3
# (confidently predicted as tomato spider mites). The model was trained by
# AgriPredict on its own corpus, and generalizes poorly to healthy foliage in
# particular — the strongest argument for the retraining pipeline in training/.
KNOWN_WEAK = [
    ("tomato_healthy", "Tomato___healthy"),
    ("corn_healthy", "Corn_(maize)___healthy"),
    ("corn_gray_leaf_spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
    ("soybean_healthy", "Soybean___healthy"),
]


@pytest.fixture(scope="module")
def detector():
    from ml_module import DiseaseDetector

    det = DiseaseDetector()
    assert det.setup(), "model failed to load"
    return det


def images_in(folder: str):
    directory = REPO / "test_images" / folder
    return sorted(p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def majority_hits(detector, folder: str, expected: str) -> tuple[int, int]:
    images = images_in(folder)
    assert images, f"no test images in {folder}"
    hits = 0
    for path in images:
        img = cv2.imread(str(path))
        assert img is not None, f"unreadable image {path}"
        preds = detector.predict_top_n(img, n=1)
        if preds and preds[0][0] == expected:
            hits += 1
    return hits, len(images)


@pytest.mark.parametrize("folder,expected", GOLDEN)
def test_golden_prediction(detector, folder, expected):
    """Each known sample folder must produce its known diagnosis as top-1
    for the majority of its images (individual shots may vary)."""
    hits, total = majority_hits(detector, folder, expected)
    assert hits >= (total + 1) // 2, (
        f"{folder}: only {hits}/{total} images predicted as {expected}"
    )


@pytest.mark.parametrize("folder,expected", KNOWN_WEAK)
@pytest.mark.xfail(strict=False, reason="known weakness of the pretrained AgriPredict model")
def test_known_weak_classes(detector, folder, expected):
    """Documents classes the current model cannot handle. An XPASS here is good
    news: it means a model change fixed the class, and it should be promoted
    into GOLDEN."""
    hits, total = majority_hits(detector, folder, expected)
    assert hits >= (total + 1) // 2, (
        f"{folder}: only {hits}/{total} images predicted as {expected}"
    )


def test_prediction_confidences_are_probabilities(detector):
    img = cv2.imread(str(images_in("corn_common_rust")[0]))
    for label, conf in detector.predict(img):
        assert 0.0 <= conf <= 1.0
        assert label in detector.labels


def test_probability_vector_covers_every_label(detector):
    """predict_probabilities() returns the full distribution, unfiltered."""
    img = cv2.imread(str(images_in("corn_common_rust")[0]))
    probs, inference_ms = detector.predict_probabilities(img)
    assert len(probs) == len(detector.labels)
    assert all(0.0 <= float(p) <= 1.0 for p in probs)
    assert inference_ms > 0


def test_analyze_pairs_predictions_with_quality_signals(detector):
    """analyze() is the desktop counterpart of the browser's guarded classify."""
    img = cv2.imread(str(images_in("corn_common_rust")[0]))
    report = detector.analyze(img, n=3)
    assert len(report["predictions"]) == 3
    assert report["predictions"][0][0] == "Corn_(maize)___Common_rust_"
    assert report["trustworthy"] is True
    assert report["capture_ok"] is True
    assert report["leaf_score"] > 0.12
    assert not report["warnings"]


def test_analyze_flags_a_non_leaf_image(detector):
    negatives = REPO / "test_images" / "_negatives" / "sky.jpg"
    if not negatives.exists():
        pytest.skip("negatives missing — run python3 tests/make_negatives.py")
    report = detector.analyze(cv2.imread(str(negatives)))
    assert report["not_leaf"] is True
    assert report["trustworthy"] is False
    assert report["warnings"]
