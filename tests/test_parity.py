"""Cross-runtime parity tests: the TFLite model and its ONNX conversion
must produce identical outputs for identical inputs.

The README claims the browser model (ONNX) is "the exact same network" as the
desktop one (TFLite). This suite enforces that claim: both runtimes are fed
byte-identical uint8 tensors and their raw quantized outputs must match
exactly. A regenerated/re-converted model that drifts fails CI.

Requires a TFLite runtime and onnxruntime; skipped automatically otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

cv2 = pytest.importorskip("cv2")
ort = pytest.importorskip("onnxruntime")

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore[no-redef]
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
        except ImportError:
            pytest.skip("no TFLite runtime installed", allow_module_level=True)

INPUT_SIZE = 300


@pytest.fixture(scope="module")
def tflite():
    interp = Interpreter(model_path=str(REPO / "models" / "plant_disease_model.tflite"))
    interp.allocate_tensors()
    return interp


@pytest.fixture(scope="module")
def onnx_session():
    return ort.InferenceSession(str(REPO / "docs" / "model" / "leafmedic.onnx"))


def run_tflite(interp, x: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    return interp.get_tensor(out["index"])[0]


def run_onnx(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0][0]


def all_test_images():
    images = sorted((REPO / "test_images").glob("*/*"))
    return [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def test_io_signatures_match(tflite, onnx_session):
    """Both models must declare the same tensor shapes and uint8 dtype."""
    t_in = tflite.get_input_details()[0]
    t_out = tflite.get_output_details()[0]
    o_in = onnx_session.get_inputs()[0]
    o_out = onnx_session.get_outputs()[0]
    assert list(t_in["shape"]) == list(o_in.shape) == [1, INPUT_SIZE, INPUT_SIZE, 3]
    assert t_in["dtype"] == np.uint8 and o_in.type == "tensor(uint8)"
    assert list(t_out["shape"]) == list(o_out.shape)
    assert t_out["dtype"] == np.uint8 and o_out.type == "tensor(uint8)"


@pytest.mark.parametrize("path", all_test_images(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_outputs_identical_on_sample_images(tflite, onnx_session, path):
    img = cv2.imread(str(path))
    assert img is not None, f"unreadable image {path}"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.uint8)[None]
    t = run_tflite(tflite, x)
    o = run_onnx(onnx_session, x)
    np.testing.assert_array_equal(
        t, o, err_msg=f"TFLite and ONNX outputs diverge on {path.name}"
    )


def test_outputs_identical_on_random_inputs(tflite, onnx_session):
    """Parity must hold across the whole input space, not just leaf photos."""
    rng = np.random.default_rng(20260718)
    for _ in range(8):
        x = rng.integers(0, 256, (1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        np.testing.assert_array_equal(run_tflite(tflite, x), run_onnx(onnx_session, x))
