"""Golden prediction tests: the model must diagnose known sample images correctly.

Requires a TFLite runtime (tensorflow, ai-edge-litert, or tflite-runtime);
skipped automatically when none is installed.
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
]


@pytest.fixture(scope="module")
def detector():
    from ml_module import DiseaseDetector

    det = DiseaseDetector()
    assert det.setup(), "model failed to load"
    return det


@pytest.mark.parametrize("folder,expected", GOLDEN)
def test_golden_prediction(detector, folder, expected):
    """Each known sample folder must produce its known diagnosis as top-1
    for the majority of its images (individual shots may vary)."""
    images = sorted((REPO / "test_images" / folder).glob("*.JPG")) + sorted(
        (REPO / "test_images" / folder).glob("*.jpg")
    )
    assert images, f"no test images in {folder}"

    hits = 0
    for path in images:
        img = cv2.imread(str(path))
        assert img is not None, f"unreadable image {path}"
        preds = detector.predict_top_n(img, n=1)
        if preds and preds[0][0] == expected:
            hits += 1
    assert hits >= (len(images) + 1) // 2, (
        f"{folder}: only {hits}/{len(images)} images predicted as {expected}"
    )


def test_prediction_confidences_are_probabilities(detector):
    img = cv2.imread(str(next((REPO / "test_images" / "corn_common_rust").glob("*.JPG"))))
    for label, conf in detector.predict(img):
        assert 0.0 <= conf <= 1.0
        assert label in detector.labels
