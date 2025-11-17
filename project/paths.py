from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).resolve().parent

# Cache directories
CACHE_DIR = REPO_ROOT / ".cache"
FIGURES_DIR = CACHE_DIR / "figures"
METRICS_DIR = CACHE_DIR / "metrics"
MODELS_DIR = CACHE_DIR / "models"
PIPELINE_DIR = CACHE_DIR / "svm_pipeline"
TENSORBOARD_DIR = CACHE_DIR / "tensorboard"

# Ensure dir
for directory in [
    CACHE_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PIPELINE_DIR,
    TENSORBOARD_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
