# Fisher-Vector SVM Pipeline

## Overview
- Dataset: CIFAR-10 training split (`50 000` RGB images, 32×32).
- Feature encoding: dense 8×8 RGB patches → PCA → 64-component diagonal GMM → improved Fisher Vectors.
- Classifier: `StandardScaler` + `LinearSVC` (`C = 1e-4`, `dual=False`), trained on Fisher Vectors.
- Random seed: `42` (shared across PCA/GMM/SVM).

## Stage 1 – Patch Extraction (`svm.extract_patches`)
- Command: `uv run python -m svm.extract_patches`
- Patches: 8×8 with stride 4 (overlapping); 7×7 grid → **49 patches per image**.
- Raw descriptor dimensionality: `8 × 8 × 3 = 192`.
- Total patches: about **2.45 M** (50 000 images × 49 patches).
- Normalization: images converted to float32 in `[0, 1]`.
- Outputs:
  - `.cache/svm_pipeline/patches.pkl` – list of per-image patch arrays (float32).
  - `.cache/svm_pipeline/labels.pkl` – CIFAR-10 labels.
  - TensorBoard logs: `.cache/tensorboard/svm/train/task_*` (training summary).

## Stage 2 – PCA Fit (`svm.train_pca`)
- Command: `uv run python -m svm.train_pca`
- PCA: `IncrementalPCA` with whitening, `PCA_DIM = 24`.
- Batch size: up to 40 000 patches per partial fit (auto-tuned).
- Explained variance retained: ≈ **96 %** (`0.96`).
- Outputs:
  - `.cache/models/svm_pca.pkl` – PCA object plus metadata.

## Stage 3 – PCA Transform (`svm.transform_patches_pca`)
- Command: `uv run python -m svm.transform_patches_pca`
- Applies saved PCA to every patch batch.
- Resulting descriptor: **24-D** per patch; still ~2.45 M vectors overall.
- Output:
  - `.cache/svm_pipeline/patches_pca.pkl`.

## Stage 4 – GMM & Fisher Vectors (`svm.compute_fisher_vectors`)
- Command: `uv run python -m svm.compute_fisher_vectors`
- GMM: `sklearn.mixture.GaussianMixture` with
  - `n_components = 64`, `covariance_type = "diag"`,
  - `max_iter = 500`, `tol = 1e-3`, `reg_covar = 1e-6`,
  - `init_params = "kmeans"`, `n_init = 1`, `random_state = 42`.
- Training set size: ~2.45 M PCA descriptors (24-D).
- Diagnostics logged: total log-likelihood, AIC, BIC, parameter count.
- Fisher Vectors: `skimage.feature.fisher_vector(..., improved=True)` per image.
  - Final FV dimension: `2 × PCA_DIM × N_COMPONENTS = 3072`.
- Outputs:
  - `.cache/svm_pipeline/gmm.pkl` – serialized sklearn GMM.
  - `.cache/svm_pipeline/fisher_vectors.pkl` – `(50 000, 3072)` float32 matrix.

## Stage 5 – Linear SVM (`svm.train`)
- Command: `uv run python -m svm.train`
- Model wrapper: `ClassifierSVM` (pipes data through scaler + LinearSVC).
- Training set: Fisher Vectors (50 000 × 3072) with CIFAR-10 labels.
- Outputs:
  - `.cache/models/svm_classifier.pkl` – bundle of PCA, GMM, and trained classifier.
  - TensorBoard logs: `.cache/tensorboard/svm/train/task_*` (training summary).

## Runtime & Resource Notes
- Patch extraction & serialization: minutes (CPU + disk bound).
- Incremental PCA: additional minutes; scales with disk throughput.
- GMM fit: dominant cost; tens of minutes on a multi-core CPU (no GPU support).
- Fisher Vector computation: a few minutes once GMM is trained.
- Linear SVM training: typically under a minute after features are cached.
- Pipeline is restartable: cached artifacts allow skipping completed stages.

## Key Hyperparameters (from `svm/constants.py`)
- `PATCH_SIZE = 8`, `STRIDE = 4`
- `PCA_DIM = 24`
- `N_COMPONENTS = 64`
- `SVM_C = 1e-4`
- `RANDOM_STATE = 42`
- Cache Root: `.cache/svm_pipeline/`

## Inference Pathway (via `ClassifierSVM`)
1. Normalize input image to float32 in `[0, 1]`.
2. Extract 8×8 patches (stride 4) → 49 descriptors.
3. Apply saved PCA transform → 24-D descriptors.
4. Encode with Fisher Vector (3072-D).
5. Standardize features and classify with linear SVM.

## Evaluation Snapshot (`uv run python -m svm.eval`)
- Dataset: CIFAR-10 test split (10 000 images).
- Accuracy: **57.30 %** (5 730 / 10 000 correct).
- Latency: total 4.65 s; prediction time 4.36 s → **0.46 ms per sample**.
- Per-class recalls:
  - airplane 58.0 %
  - automobile 67.3 %
  - bird 40.8 %
  - cat 34.1 %
  - deer 48.9 %
  - dog 52.6 %
  - frog 71.4 %
  - horse 60.9 %
  - ship 73.8 %
  - truck 65.2 %

## Benchmark Snapshot (`uv run python -m svm.benchmark`)
- Logs metrics to `.cache/tensorboard/svm/benchmark/task_*`.

