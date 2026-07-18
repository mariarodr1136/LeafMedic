# Training a LeafMedic model

The model LeafMedic currently ships is [AgriPredict's pretrained
classifier](https://www.kaggle.com/models/agripredict/disease-classification) —
16 classes, already quantized. It is good at the diseases it knows and
measurably weak elsewhere. This directory is the pipeline for replacing it with
a model trained here, end to end.

## Why replace it

`tests/test_ml.py` splits the golden tests into `GOLDEN` and `KNOWN_WEAK`, and
the second list is the argument for retraining. Measured on PlantVillage
imagery, three images per class:

| Class | Top-1 correct | What it predicts instead |
|---|---|---|
| Tomato — Septoria leaf spot | 3/3 | — |
| Tomato — Late blight | 3/3 | — |
| Tomato — Spider mites | 3/3 | — |
| Tomato — Yellow leaf curl virus | 3/3 | — |
| Corn — Common rust | 3/3 | — |
| Tomato — healthy | 1/3 | Spider mites, Septoria leaf spot |
| Corn — Gray leaf spot | 0/3 | Common rust, Tomato bacterial spot |
| Corn — healthy | 0/3 | Gray leaf spot, Tomato late blight |
| Soybean — healthy | 0/3 | Tomato spider mites (98% confidence) |

The pattern is that **healthy foliage is the weak spot**, and healthy leaves are
exactly what a user photographs when they want reassurance. The model was
trained on a different corpus than PlantVillage, so some of this is domain
shift rather than pure model quality — but either way, a model trained on the
data we actually evaluate against is the fix.

There is a second reason: `data/treatments.json` already describes **44
diseases**, while the model predicts **16**. PlantVillage's 38 classes close
most of that gap without writing any new care guidance.

## Pipeline

```bash
pip install -e '.[train]'

python3 training/download_dataset.py --per-class 200   # ~1 GB, or 0 for all
python3 training/train.py --epochs 15 --fine-tune-epochs 8
python3 training/evaluate.py --model training/runs/<run>/model_int8.tflite
```

Each run writes to `training/runs/<timestamp>-<backbone>/`:

| Artifact | What it is |
|---|---|
| `model_float.keras` | The trained float model, before quantization |
| `model_int8.tflite` | Full-integer quantized, uint8 in and out — what ships |
| `model_int8.onnx` | The same graph converted for the browser |
| `labels.txt` | Class names, in model output order |
| `metrics.json` | Test accuracy and artifact sizes |
| `history.csv` | Per-epoch loss and accuracy |
| `confusion_matrix.png` | Row-normalized, written by `evaluate.py` |
| `evaluation.json` | Per-class precision, recall, and F1 |

## Design decisions worth knowing

**Preprocessing lives inside the graph.** Rescaling is a layer, not something
the caller does. That is what lets `ml_module.py` and `docs/js/inference.js`
both feed raw uint8 RGB with no normalization — and what makes
`tests/test_parity.py` able to assert the two runtimes agree bit for bit. If
you move normalization out of the model, you must change both runtimes and
that test together.

**Bilinear resize, everywhere.** Training uses `interpolation="bilinear"` to
match `cv2.resize` on the desktop and `ctx.drawImage` in the browser. A
mismatch here is the classic silent accuracy loss: everything runs, nothing
errors, the model is just quietly fed different pixels than it was trained on.

**Quantization needs real images.** `representative_dataset` draws from the
training set. Activation ranges cannot be derived from weights, so calibrating
on noise produces a model that loads fine and predicts badly.

**Two-phase fine-tuning.** Phase 1 trains only the head with the backbone
frozen; phase 2 unfreezes the top 30% at a 100× lower learning rate. Training
everything at full learning rate from the start destroys the pretrained
features — the usual cause of a transfer-learning run that underperforms
training from scratch.

**Parity is checked at export time**, not just in CI. `train.py` runs the
TFLite and ONNX graphs against each other before you ever copy them into the
repo. See the open issue below — this check currently *fails* for models this
pipeline produces, and that is a genuine finding rather than a flaky test.

## Open issue: ONNX conversion is not bit-exact for new models

The shipped AgriPredict model is bit-identical across TFLite and ONNX —
`tests/test_parity.py` asserts exact equality and passes. Models produced by
this pipeline are **not**. Measured on 100 held-out images (deliberately
undertrained smoke runs, 300×300, tf2onnx opset 17):

| Backbone | Identical outputs | Same predicted class | Largest deviation |
|---|---|---|---|
| Shipped AgriPredict model | 100% | 100% | 0 steps |
| MobileNetV2 | 0/100 | 86–92/100 | 4–6 steps of 255 |
| MobileNetV3-Small | 0/100 | 61/100 | 18 steps of 255 |

The MobileNetV2 range is across two runs — the deviation is a property of the
trained weights, not a fixed constant per architecture, so measure your own
run rather than assuming these numbers transfer.

Two things are going on, and they should not be conflated:

1. **Numeric deviation** is the architecture-dependent part. MobileNetV3's
   hard-swish activations convert worst; MobileNetV2 is roughly 3× better.
   This is a property of the conversion and is the number `verify_parity`
   gates on (`--parity-tolerance`, default 4 steps).
2. **Argmax disagreement** in the table above is *exaggerated* by these being
   near-random models. With 38 nearly-tied scores, a one-step difference flips
   the winner. A properly trained model with confident outputs would absorb
   the same numeric noise. Do not read 61/100 as "the browser will be wrong
   39% of the time" — re-measure on a real run before concluding anything.

**Before deploying a retrained model to the browser**, run the parity check on
the actual trained model, not a smoke run. If deviation stays above a couple
of steps, prefer `--backbone mobilenetv2`, and investigate the tf2onnx opset
before shipping — otherwise the browser and desktop apps will occasionally
disagree about the same photo, which is precisely the property this project
claims to guarantee.

`tests/test_parity.py` will also start failing, because it asserts exact
equality for the shipped model. That assertion is correct today; if you deploy
a model that is merely close, relax it to the same argmax-plus-tolerance
criterion `verify_parity` uses, and say so in the README rather than quietly
loosening the claim.

## Deploying a new model

```bash
RUN=training/runs/20260718-120000-mobilenetv3

cp $RUN/model_int8.tflite models/plant_disease_model.tflite
cp $RUN/labels.txt        models/labels.txt
cp $RUN/model_int8.onnx   docs/model/leafmedic.onnx
cp $RUN/labels.txt        docs/data/labels.json   # convert to a JSON array first

python3 -m pytest tests/ -v
node tests/web_smoke.mjs
```

Three things will fail if the swap is incomplete, by design:

- `test_labels_file_matches_web_labels` — desktop and web label lists diverged
- `test_every_model_label_has_treatment_entry` — a new class has no care
  guidance in `data/treatments.json`, so it would render a diagnosis with no
  advice
- `test_outputs_identical_on_sample_images` — the ONNX conversion drifted from
  the TFLite source

Expanding to 38 classes means adding the missing entries to
`data/treatments.json` **and** its translations, then re-running
`python3 tools/build_translations.py`. `tests/test_quality.py` checks that
every translation covers the full label set, so a partial translation fails CI
rather than shipping a half-Spanish disease library.

## Hardware notes

A full 38-class run on ~54,000 images is roughly 2–4 hours on a single modern
GPU, or overnight on CPU.

**Memory is the binding constraint, not speed.** Images stream from disk by
default because caching the decoded dataset in RAM costs ~270 KB per 300×300
image — about 14 GB for the full set, which will OOM a 8–16 GB laptop partway
through a run. Pass `--cache` only if you have memory to spare; it is a solid
speedup when you do.

If you have 8 GB, prefer a rented GPU or Google Colab for the full run, and
keep the local machine for the smoke run below. Lowering `--batch-size` to 16
and using `--backbone mobilenetv3_small` also helps if you want to train
locally anyway.

To validate the pipeline before committing to any of that, run it small first:

```bash
python3 training/download_dataset.py --per-class 20
python3 training/train.py --epochs 2 --fine-tune-epochs 1 --backbone mobilenetv3_small
```

Accuracy will be poor; the point is to confirm the export, conversion, and
parity check all work before spending the GPU hours.
