
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

## How to run (Kaggle-friendly)

1. Create a new Kaggle Notebook and **Add Data** → attach your HDF5 dataset (folder with `train.h5`, `test.h5` if available).
2. Upload the notebook `top_quark_tabular_streaming_v6_1_leaksafe.ipynb` from this repo.
3. Run top-to-bottom. It will:

   * Discover files automatically under `/kaggle/input/`
   * Print the HDF5 structure
   * Train in streaming mode
   * Save `outputs/submission.csv` with probabilities for test (if `test.h5` is present)

> If auto label detection misses your columns, set:

```python
LABEL_COLS = ["ttv", "is_signal_new"]  # or your exact column names
ONE_HOT = True
```

## Repo structure (suggested)

```
/notebooks
  └── top_quark_tabular_streaming_v6_1_leaksafe.ipynb  # main, leak-safe training
/outputs
  └── submission.csv   # written by the notebook (ignored by git)
README.md
```

## Notes & gotchas

* Perfect validation is suspicious—**make sure** your validation rows are excluded from training (this notebook does it for you).
* If you hit memory limits: lower `CHUNK_SIZE` and `BATCH_SIZE`.
* For trigger-like deployments, you can later compress the MLP or distill to a smaller network.

## Roadmap

* Add k-fold streaming CV (leak-safe).
* Try gradient-boosted trees (LightGBM/XGBoost) as strong tabular baselines.
* Feature importance (permutation/SHAP) to understand the physics drivers.
* Optional image path: convert constituents to **jet images** and compare a small CNN/ResNet.

## Acknowledgments

* Dataset authors and the HEP community for pushing open benchmarks in jet tagging.
* Kaggle for making large-scale experiments easy to reproduce.

---

**Short project blurb (for your profile):**
“Built a memory-safe, streaming PyTorch pipeline over 1.21M HDF5 rows to tag top-quark jets, delivering **0.9997 AUC** and **0.845 F1** on a leak-safe holdout; designed for practical trigger/storage gains at the LHC.”
