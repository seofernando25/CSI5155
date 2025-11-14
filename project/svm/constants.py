from pathlib import Path

# Hyperparameters

# 49 Patches per image
PATCH_SIZE = 8  # 196D per patch
STRIDE = 4

PCA_DIM = 24  # 0.96 explained variance
N_COMPONENTS = 64  # GMM components
SVM_C = 1e-4
RANDOM_STATE = 42

# File paths
CACHE_DIR = Path(".cache")
MODELS_DIR = CACHE_DIR / "models"
PIPELINE_DIR = CACHE_DIR / "svm_pipeline"

# Model paths
PCA_PATH = str(MODELS_DIR / "svm_pca.pkl")
SVM_CLASSIFIER_PATH = str(MODELS_DIR / "svm_classifier.pkl")

# Pipeline intermediate results paths
PATCHES_PATH = str(PIPELINE_DIR / "patches.pkl")
LABELS_PATH = str(PIPELINE_DIR / "labels.pkl")
PCA_PATCHES_PATH = str(PIPELINE_DIR / "patches_pca.pkl")
FISHER_VECTORS_PATH = str(PIPELINE_DIR / "fisher_vectors.pkl")
GMM_PATH = str(PIPELINE_DIR / "gmm.pkl")
