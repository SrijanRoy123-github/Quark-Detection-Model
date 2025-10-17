
# Top-Quark Jet Tagging — Streaming Deep Learning on HDF5

Modern colliders like the LHC produce a firehose of particle “jets.” Most are mundane; a few come from interesting decays (like top quarks) that can hint at new physics. This project builds a **memory-safe, streaming PyTorch pipeline** that learns to tell those apart directly from large **HDF5 (pandas table) data** without loading everything into RAM.

## Why this matters (real world)

Trigger systems at the LHC must decide, **in real time**, which events to keep and which to discard. Better jet tagging means:

* **Fewer boring events written to disk** → massive storage savings.
* **More rare events retained** → better chances to spot new physics.
* **Models that are deployable** (lightweight, predictable memory use) → easier integration on FPGAs/CPUs/GPUs at the trigger level.

## What’s here

* **Streaming HDF5 reader** (pandas “table”) with chunked training and evaluation.
* **Auto label detection** (works with one-hot label pairs like `['ttv','is_signal_new']`).
* **Leak-safe validation**: builds a distributed holdout and **excludes those rows from training**.
* **Imbalance-aware MLP** (GELU + BatchNorm + Dropout) with `pos_weight`, AMP, AdamW, ReduceLROnPlateau, early stopping.
* **Threshold tuning** (Youden’s J and F1) for practical operating points.

## Data

* Source: top-quark jet tagging dataset (HDF5, pandas table with 804 numeric features + 2 label columns).
* Scale: **~1.211M rows** in `train.h5`.
* Files expected: `train.h5`, `test.h5` (optional), stored as `/table/table`.

> Tip: Use **Kaggle** to attach the dataset—no need to download 10+ GB locally.

## Method (short version)

1. **Stream** HDF5 in chunks (`start/stop`) so we never hold the whole dataset in memory.
2. **Detect labels** robustly (handles single label or two-column one-hot).
3. **Fit feature scaling** (mean/std) on a spread sample across the file to avoid bias.
4. **Build a distributed validation set** (e.g., 20k rows sampled across the file) and **exclude those exact row indices** during training.
5. Train a **GELU MLP** (1024→512→256→1) with **class weighting** via `pos_weight`.
6. Track **AUC** on the holdout, reduce LR on plateaus, and early stop if needed.
7. Tune the **decision threshold** for either **F1** or **Youden’s J** depending on your goal.

## Results (leak-safe holdout)

* **AUC**: **0.9997** (best epoch on the holdout)
* **F1-optimized threshold ≈ 0.05** → **F1: 0.8446**, **Accuracy: 0.8372**, **Precision: 0.8082**, **Recall: 0.8845**
* **Youden’s J threshold ≈ 0.06** → **Accuracy: 0.8388**, **F1: 0.8440**
* Holdout size: **13,000** rows (excluded from all training), full train scale: **~1.211M** rows.

These numbers reflect a realistic, leak-safe estimate of how the model generalizes.



## Acknowledgments

* Dataset authors and the HEP community for pushing open benchmarks in jet tagging.
* Kaggle for making large-scale experiments easy to reproduce.

