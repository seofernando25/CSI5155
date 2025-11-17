from pathlib import Path
import numpy as np
from skimage.feature import fisher_vector
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Optional
from tqdm import tqdm
import joblib
from utils import require_file


class ClassifierSVM:
    def __init__(
        self,
        pca: IncrementalPCA,
        patch_size: int = 8,
        stride: int = 8,
        n_components: int = 32,
        svm_C: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.n_components = n_components
        self.svm_C = svm_C
        self.random_state = random_state

        # Set PCA object and get dimension from it
        self.pca = pca
        self.pca_dim = pca.n_components

        self.gmm = None
        self.classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=svm_C,
                        random_state=random_state,
                        dual=False,
                        verbose=1,
                    ),
                ),
            ]
        )

    def _extract_patches_2d_stride(self, image, patch_size, stride):
        patch_h = patch_w = patch_size

        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        i_h, i_w, n_channels = image.shape

        # Calculate number of patches
        n_patches_h = (i_h - patch_h) // stride + 1
        n_patches_w = (i_w - patch_w) // stride + 1
        n_patches = n_patches_h * n_patches_w

        # Pre-allocate output array
        patches = np.zeros((n_patches, patch_h, patch_w, n_channels), dtype=image.dtype)

        # Extract patches
        patch_idx = 0
        for i in range(0, i_h - patch_h + 1, stride):
            for j in range(0, i_w - patch_w + 1, stride):
                patches[patch_idx] = image[i : i + patch_h, j : j + patch_w, :]
                patch_idx += 1

        return patches

    def _extract_patches(self, images: List[np.ndarray]) -> List[np.ndarray]:
        descriptors_list = []

        for img in tqdm(images, desc="Extracting patches", leave=False):
            patches = self._extract_patches_2d_stride(img, self.patch_size, self.stride)
            # Reshape and convert to float32 for memory efficiency
            descriptors_list.append(
                patches.reshape(len(patches), -1).astype(np.float32)
            )

        return descriptors_list

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        # Images from processed dataset are already float32 in [0,1]
        X = [np.asarray(img, dtype=np.float32) for img in X]
        rgb_descs = self._extract_patches(X)

        # Compute Fisher Vectors sequentially
        fvs = []
        for img_desc in tqdm(rgb_descs, desc="Computing Fisher Vectors", leave=False):
            desc_pca = self.pca.transform(img_desc)
            fv = fisher_vector(desc_pca, self.gmm, improved=True)
            fvs.append(fv)
        fvs = np.array(fvs, dtype=np.float32)

        return self.classifier.predict(fvs)

    def score(self, X: List[np.ndarray], y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def save(self, filepath: str) -> None:
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save all components
        model_data = {
            "hyperparameters": {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "n_components": self.n_components,
                "pca_dim": self.pca_dim,
                "svm_C": self.svm_C,
                "random_state": self.random_state,
            },
            "pca": self.pca,
            "gmm": self.gmm,
            "classifier": self.classifier,
        }

        # Use joblib for sklearn objects (more efficient for numpy arrays)
        if filepath_obj.suffix in [".joblib", ".pkl"]:
            joblib.dump(model_data, filepath_obj)
        else:
            # Default to .pkl if no extension
            joblib.dump(model_data, str(filepath_obj) + ".pkl")

    @classmethod
    def load(cls, filepath: str) -> "ClassifierSVM":
        filepath_obj = require_file(filepath)

        # Load model data
        model_data = joblib.load(filepath_obj)

        # Recreate model with saved hyperparameters
        hyperparams = model_data["hyperparameters"]
        # Handle backward compatibility: old models may not have stride
        stride = hyperparams.get("stride", 8)  # Default to 8 for non-overlapping
        # PCA is loaded from saved model, pca_dim will be set from it
        model = cls(
            pca=model_data["pca"],  # Load PCA first
            patch_size=hyperparams["patch_size"],
            stride=stride,
            n_components=hyperparams["n_components"],
            svm_C=hyperparams["svm_C"],
            random_state=hyperparams["random_state"],
        )

        # Restore trained components
        gmm_data = model_data["gmm"]

        # Handle GMM format: could be dict (with sklearn_gmm) or direct sklearn GMM
        if isinstance(gmm_data, dict):
            if "sklearn_gmm" in gmm_data:
                model.gmm = gmm_data["sklearn_gmm"]
            else:
                raise ValueError(
                    f"Invalid GMM format in saved model. Expected 'sklearn_gmm' key, got: {list(gmm_data.keys())}"
                )
        else:
            # Direct sklearn GMM object
            model.gmm = gmm_data

        model.classifier = model_data["classifier"]

        return model
