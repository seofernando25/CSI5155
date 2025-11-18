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

## Usage

All commands are run through the main entry point:

```bash
uv run main.py <command>
```

### SVM Pipeline

Train and evaluate the SVM classifier on CIFAR-10:

```bash
# Extract patches
uv run main.py svm extract-patches
# Train PCA
uv run main.py svm train-pca
# Transform patches
uv run main.py svm transform-patches
# Compute Fisher Vectors
uv run main.py svm compute-fv
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
# Train model
uv run main.py scaledcnn train --k 4
# Evaluate checkpoint
uv run main.py scaledcnn eval --k 4
# Generate training report and confusion matrix
uv run main.py scaledcnn report --k 4
# Plot capacity curve
uv run main.py scaledcnn capacity-curve
```
