#!/usr/bin/env python3

"""
LeafMedic Inference Benchmark
=============================
Measures inference latency on *this* machine so the numbers in the README can
be reproduced rather than taken on trust.

Reports the median rather than the mean: a single scheduler hiccup skews a
mean over a short run, while the median reflects what a user actually waits
for. Percentiles are included because tail latency is what makes an interface
feel slow.

Usage:
    python3 benchmark.py                    # TFLite, 50 runs
    python3 benchmark.py --runs 200
    python3 benchmark.py --onnx             # also benchmark ONNX Runtime
    python3 benchmark.py --image path.jpg   # use a specific image
    python3 benchmark.py --json             # machine-readable output

The browser side has an equivalent: open the demo with ?bench, or call
`leafmedicBenchmark(50)` from the console.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
INPUT_SIZE = (300, 300)
DEFAULT_IMAGE = REPO / "test_images" / "corn_common_rust"


def find_sample_image(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"image not found: {path}")
        return path
    candidates = sorted(
        p for p in DEFAULT_IMAGE.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise SystemExit(f"no sample images in {DEFAULT_IMAGE}")
    return candidates[0]


def load_tensor(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"unreadable image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, INPUT_SIZE).astype(np.uint8)[None]


def summarize(times_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(times_ms)
    return {
        "runs": len(ordered),
        "median_ms": round(statistics.median(ordered), 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "min_ms": round(ordered[0], 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        "max_ms": round(ordered[-1], 2),
        "stdev_ms": round(statistics.pstdev(ordered), 2),
    }


def time_runs(run, tensor: np.ndarray, runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        run(tensor)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run(tensor)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def bench_tflite(tensor: np.ndarray, runs: int, warmup: int) -> dict[str, Any] | None:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                return None

    model = REPO / "models" / "plant_disease_model.tflite"
    if not model.exists():
        return None

    interpreter = Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    def run(x):
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        return interpreter.get_tensor(out["index"])

    result = summarize(time_runs(run, tensor, runs, warmup))
    result["runtime"] = f"TFLite ({Interpreter.__module__.split('.')[0]})"
    result["model_mb"] = round(model.stat().st_size / (1024 * 1024), 2)
    result["top_class"] = int(np.argmax(run(tensor)[0]))
    return result


def bench_onnx(tensor: np.ndarray, runs: int, warmup: int) -> dict[str, Any] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    model = REPO / "docs" / "model" / "leafmedic.onnx"
    if not model.exists():
        return None

    session = ort.InferenceSession(str(model))
    name = session.get_inputs()[0].name

    def run(x):
        return session.run(None, {name: x})[0]

    result = summarize(time_runs(run, tensor, runs, warmup))
    result["runtime"] = f"ONNX Runtime {ort.__version__}"
    result["model_mb"] = round(model.stat().st_size / (1024 * 1024), 2)
    result["top_class"] = int(np.argmax(run(tensor)[0]))
    return result


def machine_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def print_report(results: dict[str, dict[str, Any]], info: dict[str, str], image: Path) -> None:
    print("=" * 66)
    print("  LeafMedic Inference Benchmark")
    print("=" * 66)
    print(f"  Machine : {info['processor']} · {info['platform']}")
    print(f"  Python  : {info['python']}")
    print(f"  Image   : {image.name} (resized to {INPUT_SIZE[0]}x{INPUT_SIZE[1]}, uint8)")
    print("=" * 66)
    print()

    header = f"{'Runtime':<28}{'median':>9}{'min':>9}{'p95':>9}{'max':>9}"
    print(header)
    print("-" * len(header))
    for stats in results.values():
        print(
            f"{stats['runtime']:<28}"
            f"{stats['median_ms']:>8.1f}ms"
            f"{stats['min_ms']:>8.1f}ms"
            f"{stats['p95_ms']:>8.1f}ms"
            f"{stats['max_ms']:>8.1f}ms"
        )
    print()

    classes = {stats["top_class"] for stats in results.values()}
    if len(results) > 1:
        if len(classes) == 1:
            print("  ✓ All runtimes agree on the predicted class")
        else:
            print("  ✗ Runtimes DISAGREE on the predicted class — check conversion")
    print(f"  {next(iter(results.values()))['runs']} timed runs per runtime after warm-up")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LeafMedic inference latency")
    parser.add_argument("--runs", type=int, default=50, help="timed runs per runtime (default 50)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warm-up runs (default 5)")
    parser.add_argument("--image", help="image to benchmark with (default: a bundled sample)")
    parser.add_argument("--onnx", action="store_true", help="also benchmark ONNX Runtime")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    image = find_sample_image(args.image)
    tensor = load_tensor(image)

    results: dict[str, dict[str, Any]] = {}
    tflite = bench_tflite(tensor, args.runs, args.warmup)
    if tflite:
        results["tflite"] = tflite
    elif not args.json:
        print("! No TFLite runtime installed — skipping (pip install ai-edge-litert)")

    if args.onnx:
        onnx = bench_onnx(tensor, args.runs, args.warmup)
        if onnx:
            results["onnx"] = onnx
        elif not args.json:
            print("! onnxruntime not installed — skipping (pip install onnxruntime)")

    if not results:
        print("No runtime available to benchmark.", file=sys.stderr)
        return 1

    info = machine_info()
    if args.json:
        print(json.dumps({"machine": info, "image": image.name, "results": results}, indent=2))
    else:
        print_report(results, info, image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
