"""Data integrity tests: labels and the treatment knowledge base."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REQUIRED_DISEASE_FIELDS = {"common_name", "plant", "description", "symptoms", "treatments", "prevention"}


def load_labels():
    return [
        line.strip()
        for line in (REPO / "models" / "labels.txt").read_text().splitlines()
        if line.strip()
    ]


def load_treatments():
    return json.loads((REPO / "data" / "treatments.json").read_text())


def test_labels_file_matches_web_labels():
    """The desktop labels.txt and the web demo's labels.json must be identical."""
    desktop = load_labels()
    web = json.loads((REPO / "docs" / "data" / "labels.json").read_text())
    assert desktop == web


def test_every_model_label_has_treatment_entry():
    """Every class the model can predict must have care guidance."""
    treatments = load_treatments()
    missing = [label for label in load_labels() if label not in treatments]
    assert not missing, f"Labels without treatment entries: {missing}"


def test_treatment_entries_have_required_fields():
    treatments = load_treatments()
    for label, info in treatments.items():
        if "healthy" in label.lower():
            continue  # healthy entries only need a description
        missing = REQUIRED_DISEASE_FIELDS - set(info)
        assert not missing, f"{label} missing fields: {missing}"


def test_desktop_and_web_treatments_match():
    desktop = load_treatments()
    web = json.loads((REPO / "docs" / "data" / "treatments.json").read_text())
    assert desktop == web


def test_severity_values_are_known():
    known = {"low", "medium", "high", "critical", "none", None}
    for label, info in load_treatments().items():
        assert info.get("severity") in known, f"{label} has unknown severity {info.get('severity')!r}"
