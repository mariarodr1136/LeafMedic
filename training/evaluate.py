#!/usr/bin/env python3

"""
Model Evaluation & Confusion Matrix
===================================
Scores a trained/quantized model on a held-out set and writes the artifacts
that make an accuracy claim checkable: per-class precision/recall, a confusion
matrix image, and the list of classes that are actually being confused.

Works on the quantized .tflite directly, so what is measured is what ships —
not the float model it was derived from. Quantization can cost a point or two
of accuracy, and this is where that shows up.

Usage:
    python3 training/evaluate.py --model training/runs/<run>/model_int8.tflite
    python3 training/evaluate.py --model models/plant_disease_model.tflite \
        --labels models/labels.txt --data training/data/plantvillage
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IMAGE_SIZE = 300


def load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, image: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], image[None].astype(inp["dtype"]))
    interpreter.invoke()
    scores = interpreter.get_tensor(out["index"])[0]
    if scores.dtype == np.uint8:
        scores = scores.astype(np.float32) / 255.0
    return scores


def collect_predictions(interpreter, data_dir: Path, labels: list[str], limit: int):
    import cv2

    y_true: list[int] = []
    y_pred: list[int] = []
    skipped: Counter[str] = Counter()

    for class_index, label in enumerate(labels):
        class_dir = data_dir / label
        if not class_dir.exists():
            skipped[label] += 1
            continue
        paths = sorted(
            p for p in class_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if limit:
            paths = paths[:limit]
        for path in paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
            y_true.append(class_index)
            y_pred.append(int(np.argmax(predict(interpreter, resized))))

    return np.array(y_true), np.array(y_pred), skipped


def per_class_report(y_true, y_pred, labels):
    rows = []
    for index, label in enumerate(labels):
        support = int((y_true == index).sum())
        if support == 0:
            continue
        tp = int(((y_pred == index) & (y_true == index)).sum())
        fp = int(((y_pred == index) & (y_true != index)).sum())
        fn = int(((y_pred != index) & (y_true == index)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "label": label, "support": support,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
    return rows


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("! matplotlib not installed — skipping confusion matrix image")
        return None

    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    # Row-normalize so classes with more samples don't dominate the colour scale.
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
    im = ax.imshow(normalized, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("LeafMedic confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Confusion matrix: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a quantized LeafMedic model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path,
                        help="labels.txt (default: alongside the model, else models/labels.txt)")
    parser.add_argument("--data", type=Path, default=REPO / "training" / "data" / "plantvillage")
    parser.add_argument("--limit", type=int, default=100, help="images per class (0 = all)")
    parser.add_argument("--out", type=Path, help="output directory (default: next to the model)")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    labels_path = args.labels
    if labels_path is None:
        sibling = args.model.parent / "labels.txt"
        labels_path = sibling if sibling.exists() else REPO / "models" / "labels.txt"
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]

    if not args.data.exists():
        raise SystemExit(
            f"dataset not found at {args.data}\nRun: python3 training/download_dataset.py"
        )

    out_dir = args.out or args.model.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {args.model}")
    print(f"Labels : {labels_path} ({len(labels)} classes)")
    print(f"Data   : {args.data}\n")

    interpreter = load_interpreter(args.model)
    y_true, y_pred, skipped = collect_predictions(interpreter, args.data, labels, args.limit)

    if len(y_true) == 0:
        raise SystemExit(
            "no evaluable images found — do the dataset folder names match labels.txt?"
        )

    accuracy = float((y_true == y_pred).mean())
    rows = per_class_report(y_true, y_pred, labels)

    print(f"{'class':<52}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>7}")
    print("-" * 80)
    for row in sorted(rows, key=lambda r: r["recall"]):
        print(f"{row['label']:<52}{row['precision']:>7.3f}{row['recall']:>7.3f}"
              f"{row['f1']:>7.3f}{row['support']:>7}")
    print("-" * 80)
    print(f"{'overall accuracy':<52}{accuracy:>7.3f}{'':>7}{'':>7}{len(y_true):>7}")

    if skipped:
        print(f"\n! {len(skipped)} label(s) had no data directory: {', '.join(sorted(skipped))}")

    weak = [r for r in rows if r["recall"] < 0.7]
    if weak:
        print("\nWeakest classes (recall < 0.70):")
        for row in sorted(weak, key=lambda r: r["recall"]):
            print(f"  {row['label']}  recall={row['recall']:.2f}")

    save_confusion_matrix(y_true, y_pred, labels, out_dir / "confusion_matrix.png")

    report = {
        "model": str(args.model),
        "accuracy": round(accuracy, 4),
        "images": int(len(y_true)),
        "per_class": rows,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"✓ Report: {out_dir / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
