from pathlib import Path

# Repository root (directory containing this file)
REPO_ROOT = Path(__file__).resolve().parent

# Cache directories
CACHE_DIR = REPO_ROOT / ".cache"
FIGURES_DIR = CACHE_DIR / "figures"
METRICS_DIR = CACHE_DIR / "metrics"
MODELS_DIR = CACHE_DIR / "models"
TENSORBOARD_DIR = CACHE_DIR / "tensorboard"

# Ensure all directories exist
for directory in [CACHE_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, TENSORBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

