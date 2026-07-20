"""The browser fetches docs/model/leafmedic.onnx and verifies it against
docs/model/manifest.json before running inference (see verifyModelIntegrity
in docs/js/inference.js). If the model is ever regenerated without
regenerating the manifest, every browser load would fail that check — this
test catches the drift in CI instead of in a user's browser.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "docs" / "model" / "leafmedic.onnx"
MANIFEST = REPO / "docs" / "model" / "manifest.json"


def test_manifest_matches_model_bytes():
    manifest = json.loads(MANIFEST.read_text())
    data = MODEL.read_bytes()
    assert manifest["bytes"] == len(data)
    assert manifest["sha256"] == hashlib.sha256(data).hexdigest()
