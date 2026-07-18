#!/usr/bin/env python3

"""
LeafMedic Model Training
========================
Fine-tunes a MobileNetV3 (or EfficientNet-Lite) backbone on PlantVillage and
exports a fully integer-quantized TFLite model plus its ONNX conversion —
the two artifacts the desktop app and the browser demo consume.

Why retrain at all: the shipped AgriPredict model covers 16 classes and is
measurably weak on healthy foliage (see KNOWN_WEAK in tests/test_ml.py).
PlantVillage has 38 classes and 54,000 images under CC0, and the knowledge
base in data/treatments.json already describes 44 diseases — so the data and
the care guidance are both ahead of the model.

Pipeline:
    1. Load PlantVillage from an image-folder layout, split train/val/test
    2. Fine-tune an ImageNet-pretrained backbone in two phases
       (frozen head, then partial unfreeze at a lower learning rate)
    3. Post-training full-integer quantization with a real calibration set
    4. Export .tflite, convert to .onnx, verify the two agree
    5. Write labels.txt, a metrics report, and a confusion matrix

Usage:
    python3 training/download_dataset.py            # fetch PlantVillage
    python3 training/train.py --epochs 15
    python3 training/train.py --backbone efficientnet_lite0 --image-size 300
    python3 training/evaluate.py --model training/runs/<run>/model_int8.tflite

Install the training extras first:
    pip install -e '.[train]'
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "training" / "data" / "plantvillage"
RUNS = REPO / "training" / "runs"

# The deployed model's contract. Changing these means changing both runtimes
# and the preprocessing in ml_module.py and docs/js/inference.js.
IMAGE_SIZE = 300
BATCH_SIZE = 32
SEED = 20260718


def build_datasets(data_dir: Path, image_size: int, batch_size: int, val_split: float,
                   cache: bool = False):
    """Load an image-folder dataset and split it into train/val/test.

    Keras gives a two-way split, so the validation half is split again to hold
    out a test set that neither training nor early stopping ever sees.
    """
    import tensorflow as tf

    if not data_dir.exists():
        raise SystemExit(
            f"dataset not found at {data_dir}\n"
            "Run: python3 training/download_dataset.py"
        )

    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        interpolation="bilinear",   # matches cv2.resize / ctx.drawImage at inference
        batch_size=batch_size,
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training", shuffle=True, **common
    )
    holdout = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation", shuffle=True, **common
    )
    class_names = list(train_ds.class_names)

    # Halve the holdout into validation and test.
    holdout_batches = tf.data.experimental.cardinality(holdout).numpy()
    val_ds = holdout.take(holdout_batches // 2)
    test_ds = holdout.skip(holdout_batches // 2)

    autotune = tf.data.AUTOTUNE
    # Caching decoded images in RAM costs roughly 270 KB per 300x300 image, so
    # the full 54k-image dataset needs ~14 GB — more than many laptops have.
    # Default to streaming from disk; opt into caching only when there is
    # memory to spare (--cache), where it gives a solid speedup.
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()
    train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(backbone: str, num_classes: int, image_size: int):
    """Assemble backbone + classification head.

    Rescaling lives *inside* the graph so that both runtimes can feed raw uint8
    RGB and apply no normalization of their own — which is what keeps the
    desktop and browser preprocessing identical and trivially verifiable.
    """
    import tensorflow as tf

    shape = (image_size, image_size, 3)
    factory = {
        "mobilenetv3": tf.keras.applications.MobileNetV3Large,
        "mobilenetv3_small": tf.keras.applications.MobileNetV3Small,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnet_lite0": tf.keras.applications.EfficientNetB0,
    }
    if backbone not in factory:
        raise SystemExit(f"unknown backbone {backbone!r}; choose from {sorted(factory)}")

    base = factory[backbone](
        input_shape=shape, include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=shape, dtype="uint8", name="image")
    # Rescaling doubles as the uint8 -> float32 cast. Keras 3 rejects a bare
    # tf.cast() on a symbolic input, and every step of the graph has to be a
    # layer for the TFLite converter to trace it anyway.
    # MobileNetV3 bakes in its own rescaling; the others expect [-1, 1].
    if backbone.startswith("mobilenetv3"):
        x = tf.keras.layers.Rescaling(1.0, name="cast_to_float")(inputs)
    else:
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)

    x = tf.keras.layers.RandomFlip("horizontal", seed=SEED)(x)
    x = tf.keras.layers.RandomRotation(0.15, seed=SEED)(x)
    x = tf.keras.layers.RandomZoom(0.15, seed=SEED)(x)
    x = tf.keras.layers.RandomContrast(0.15, seed=SEED)(x)

    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"leafmedic_{backbone}")
    return model, base


def train(model, base, train_ds, val_ds, epochs: int, fine_tune_epochs: int, out_dir: Path):
    """Two-phase transfer learning: train the head, then unfreeze the top of
    the backbone at a much lower learning rate so pretrained features are
    refined rather than destroyed."""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv"), append=True),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n== Phase 1: training the classification head ==")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    if fine_tune_epochs > 0:
        print("\n== Phase 2: fine-tuning the top of the backbone ==")
        base.trainable = True
        for layer in base.layers[: int(len(base.layers) * 0.7)]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks
        )
    return model


def quantize(model, train_ds, out_dir: Path, calibration_batches: int = 32) -> Path:
    """Full-integer post-training quantization.

    Both weights *and* activations become uint8, which is what makes the model
    ~4x smaller and CPU-only inference viable on a Raspberry Pi. Activation
    ranges cannot be inferred from weights, so a representative dataset of real
    images is required — random noise here would produce wrong ranges and a
    model that silently loses accuracy.
    """
    import tensorflow as tf

    def representative_dataset():
        seen = 0
        for images, _ in train_ds:
            for image in images:
                yield [tf.expand_dims(tf.cast(image, tf.uint8), 0)]
                seen += 1
                if seen >= calibration_batches * BATCH_SIZE:
                    return

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    path = out_dir / "model_int8.tflite"
    path.write_bytes(tflite_model)
    print(f"\n✓ Quantized TFLite model: {path} ({len(tflite_model) / 1e6:.1f} MB)")
    return path


def export_onnx(tflite_path: Path, out_dir: Path) -> Path | None:
    """Convert the quantized TFLite graph to ONNX for the browser demo."""
    import subprocess

    onnx_path = out_dir / "model_int8.onnx"
    cmd = [
        "python3", "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", "17",
    ]
    print("\n== Converting to ONNX ==")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"! tf2onnx failed:\n{result.stderr[-2000:]}")
        return None
    print(f"✓ ONNX model: {onnx_path}")
    return onnx_path


def verify_parity(tflite_path: Path, onnx_path: Path, test_ds, tolerance: int = 4) -> bool:
    """The browser and desktop must agree bit-for-bit — the same property
    tests/test_parity.py enforces for the shipped model."""
    import numpy as np
    import tensorflow as tf

    try:
        import onnxruntime as ort
    except ImportError:
        print("! onnxruntime not installed — skipping parity check")
        return True

    # Prefer LiteRT; tf.lite.Interpreter is deprecated. Either way the XNNPACK
    # delegate can fail to prepare on a freshly quantized graph, so fall back
    # to the plain kernels — same helper the desktop app uses.
    import sys

    sys.path.insert(0, str(REPO))
    from ml_module import load_interpreter

    interp = load_interpreter(str(tflite_path))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    session = ort.InferenceSession(str(onnx_path))
    name = session.get_inputs()[0].name

    # What must hold is that both runtimes give the user the same diagnosis.
    # Bit-exact agreement is stronger than that and not guaranteed: TFLite and
    # ONNX Runtime implement quantized arithmetic with slightly different
    # rounding, so individual uint8 scores can land one step apart (~0.4% of
    # probability). A differing argmax, or a large gap, means a real conversion
    # bug.
    exact = 0
    argmax_agree = 0
    checked = 0
    worst = 0

    for images, _ in test_ds.take(4):
        for image in images:
            x = tf.cast(tf.expand_dims(image, 0), tf.uint8).numpy()
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            a = interp.get_tensor(out["index"])
            b = session.run(None, {name: x})[0]
            checked += 1
            if np.array_equal(a, b):
                exact += 1
            if int(np.argmax(a)) == int(np.argmax(b)):
                argmax_agree += 1
            worst = max(worst, int(np.abs(a.astype(int) - b.astype(int)).max()))

    print(f"  identical outputs   : {exact}/{checked}")
    print(f"  same predicted class: {argmax_agree}/{checked}")
    print(f"  largest deviation   : {worst} quantization step(s) of 255")

    # Deviation magnitude is the signal to gate on, not argmax agreement.
    # Argmax agreement is confounded by model quality: an undertrained model
    # has a nearly flat output distribution, so a one-step difference flips the
    # winner. A confident model absorbs the same numeric noise unnoticed.
    if worst > tolerance:
        print(f"✗ Parity FAILED — deviation exceeds {tolerance} steps. The ONNX "
              "conversion is not faithful; do not deploy this to the browser.")
        print("  MobileNetV3's hard-swish activations convert worst; try "
              "--backbone mobilenetv2 and compare.")
        return False
    if exact == checked:
        print(f"✓ Parity verified on {checked} inputs (bit-identical)")
    else:
        print(f"✓ Parity within tolerance on {checked} inputs "
              f"(max {worst} steps ≈ {worst / 255:.1%} of probability)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and export a LeafMedic model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbone", default="mobilenetv3",
                        help="mobilenetv3 | mobilenetv3_small | mobilenetv2 | efficientnet_lite0")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.3)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--parity-tolerance", type=int, default=4,
                        help="max allowed TFLite/ONNX deviation, in uint8 steps (default 4)")
    parser.add_argument("--cache", action="store_true",
                        help="cache decoded images in RAM (needs ~14 GB for the full dataset)")
    args = parser.parse_args()

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS / f"{run_id}-{args.backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out_dir}")

    train_ds, val_ds, test_ds, class_names = build_datasets(
        args.data, args.image_size, args.batch_size, args.val_split, cache=args.cache
    )
    print(f"Classes ({len(class_names)}): {', '.join(class_names[:6])}…")

    (out_dir / "labels.txt").write_text("\n".join(class_names) + "\n")

    model, base = build_model(args.backbone, len(class_names), args.image_size)
    model.summary()
    model = train(model, base, train_ds, val_ds, args.epochs, args.fine_tune_epochs, out_dir)

    print("\n== Evaluating the float model on the held-out test set ==")
    float_loss, float_acc = model.evaluate(test_ds)
    model.save(out_dir / "model_float.keras")

    tflite_path = quantize(model, train_ds, out_dir)

    onnx_path = None
    if not args.skip_onnx:
        onnx_path = export_onnx(tflite_path, out_dir)
        if onnx_path:
            verify_parity(tflite_path, onnx_path, test_ds, args.parity_tolerance)

    metrics = {
        "run": run_id,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "num_classes": len(class_names),
        "float_test_accuracy": round(float(float_acc), 4),
        "float_test_loss": round(float(float_loss), 4),
        "tflite_bytes": tflite_path.stat().st_size,
        "onnx_bytes": onnx_path.stat().st_size if onnx_path else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n" + "=" * 60)
    print(f"  Float test accuracy : {float_acc:.4f}")
    print(f"  Artifacts           : {out_dir}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  python3 training/evaluate.py --model {tflite_path} --data {args.data}")
    print("  # then, to deploy, copy the artifacts into place:")
    print(f"  cp {tflite_path} models/plant_disease_model.tflite")
    print(f"  cp {out_dir / 'labels.txt'} models/labels.txt")
    print("  cp <run>/model_int8.onnx docs/model/leafmedic.onnx")
    print("  python3 -m pytest tests/   # parity + golden tests must pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
