from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture

from utils.paths import require_file


def load_pca(path: Path | str) -> IncrementalPCA:
    path_obj = require_file(path)
    pca_data = joblib.load(str(path_obj))
    pca = pca_data["pca"]
    assert isinstance(pca, IncrementalPCA), f"Expected IncrementalPCA, got {type(pca).__name__}"
    return pca


def load_gmm(path: Path | str) -> GaussianMixture:
    path_obj = require_file(path)
    gmm_data = joblib.load(str(path_obj))
    gmm = gmm_data["sklearn_gmm"]
    assert isinstance(gmm, GaussianMixture), f"Expected GaussianMixture, got {type(gmm).__name__}"
    return gmm

