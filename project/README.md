# From Handcrafted Features to Double Descent on CIFAR-10

CSI 5155 - Machine Learning

Project Report (Fall 2025)

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

Install dependencies:

```bash
uv sync
```

## Quick Start

**Important:** Run data processing first before either pipeline.

1. Process data: `uv run main.py data download` then `uv run main.py data process`
2. Choose a pipeline (SVM or ScaledCNN) and run commands in order, see usage below
3. Results are saved to `.cache/` (figures, models, logs, etc)


## Usage

All commands are run through the main entry point:

```bash
uv run main.py <command>
```

### Data Processing

Download and prepare datasets (required before running pipelines)

```bash
# Download CIFAR-10 dataset
uv run main.py data download
# Process datasets (normalize, split train/val, add label noise)
uv run main.py data process
# Launch dataset explorer server (optional)
uv run main.py data explorer
```

### SVM Pipeline

Train and evaluate the SVM classifier (run commands in order):

```bash
# Extract patches
uv run main.py svm extract-patches
# Train PCA
uv run main.py svm train-pca
# Transform patches
uv run main.py svm transform-patches
# Compute Fisher Vectors
uv run main.py svm compute-fv
# Hyperparameter Optimization (Optional, tuned constants already in constants.py)
uv run main.py svm hparam
# Train SVM
uv run main.py svm train
# Evaluate
uv run main.py svm eval
# Generate report and confusion matrix
uv run main.py svm report
```

### ScaledCNN

Train and evaluate ScaledCNN models:

```bash
# Train model (k is the width scaling factor: 1, 2, 4, 8, 16, 32, 64)
uv run main.py scaledcnn train --k 4
# Evaluate checkpoint
uv run main.py scaledcnn eval --k 4
# Generate training report and confusion matrix
uv run main.py scaledcnn report --k 4
# Plot capacity curve (after training with multiple k values)
uv run main.py scaledcnn capacity-curve
```

Training logs are available in TensorBoard: `tensorboard --logdir .cache/tensorboard/training/[experiment_name]`

