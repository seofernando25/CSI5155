### CSI5155 Assignment 2 — Presentation Order

Use this as a concise run-of-show for the TA demo. Each step includes what to show and the exact command to run if applicable.

---

## 1) Project Overview (1 min)
- **Goal**: Evaluate semi-supervised learning on the UCI Optical Digits dataset; compare `SelfTrainingClassifier` and `LabelPropagation` to a supervised SVM baseline.
- **Repo layout**:
  - `scripts/`: entrypoints (`pca_analysis.py`, `baseline_svm.py`, `self_training.py`, `label_propagation.py`, `plot_results.py`, `validate_report.py`)
  - `lib/`: reusable utilities (`data_utils.py`, `semi_supervised_utils.py`, `plot_utils.py`)
  - `.cache/`: dataset, results, and generated figures (tracked for reproducibility in this repo)
  - `REPORT.md`: written analysis; `AS2.md`: assignment brief; `Makefile`: handy commands

## 2) Environment and Setup (30 sec)
- Python 3.12, dependencies in `pyproject.toml`.
- Install once:
```bash
make install
```

## 3) Dataset (30 sec)
- Stored under `.cache/dataset/` and loaded via `lib/data_utils.py`.
- Show where it is defined and used (`DATA_DIR`, `load_full_dataset`).
- If needed, (re)download and extract:
```bash
make download
```

## 4) PCA Analysis — Smoothness Assumption (1–2 min)
- What to say: PCA to 2D shows clusters by digit; silhouette in 64D (0.33) > 2D (0.27) but cluster structure persists.
- Run and show saved figures:
```bash
make pca
```
- Open figures: `.cache/figures/pca_analysis/pca_scatter.png` and `pca_grid.png`.

## 5) Supervised Baseline (SVM) (1 min)
- What to say: 70/30 split; linear SVM with C=0.01; baseline macro F1 and accuracy ~0.98.
- Run:
```bash
make baseline_svm
```
- Output saved to `.cache/results/baseline.json`.

## 6) Semi-Supervised Experiments (3–4 min)
- What to say: Evaluate 12 proportions p ∈ {0.2%, …, 10%}; 5 runs each; compare ST vs LP.
- Self-Training (SVM base, `criterion='k_best'`, `max_iter=100`):
```bash
make self-training
```
- Label Propagation (`kernel='knn'`, `n_neighbors=7`):
```bash
make label-propagation
```
- Notes: Mask labels to -1 using `label_mask`; train on semi-supervised training set; metrics via `classification_report`.

## 7) Plots and Quantitative Summary (2–3 min)
- Generate comparison plots:
```bash
make plots
```
- Show figures in `.cache/figures/semi_supervised/`:
  - `accuracy.png`, `macro_f1.png`, `macro_precision.png`, `macro_recall.png`
  - Per-class F1: `f1_class_6.png`, `f1_class_8.png`
- Optional tabular/insight summary (quick CLI output):
```bash
uv run python -m scripts.validate_report
```

## 8) Key Findings (1–2 min)
- From `REPORT.md`:
  - LP > ST across metrics and label proportions; at p=10%, LP ≈ baseline (macro F1 ~0.98).
  - ST struggles at very low p on class 8; LP needs more labels to stabilize but wins overall.
  - Improvements possible via hyperparameter tuning, calibration for ST, or different base models.

## 9) Files to Highlight (30–45 sec)
- `lib/semi_supervised_utils.py`: proportions, masking, experiment loop, saving results.
- `lib/data_utils.py`: dataset paths, split, result I/O.
- `scripts/plot_results.py` + `lib/plot_utils.py`: standardized plotting and saving.

## 10) Cleanup (optional)
- Remove generated outputs (if needed for a clean rerun):
```bash
make clean
```

---

### Suggested Timing (~10–12 minutes)
- Overview: 1 min
- Setup + dataset: 1 min
- PCA: 2 min
- Baseline: 1 min
- SSL experiments: 4 min
- Plots + findings: 2–3 min

### Quick Demo Flow (commands only)
```bash
make install
make download
make baseline_svm
make pca
make self-training
make label-propagation
make plots
uv run python -m scripts.validate_report
```


