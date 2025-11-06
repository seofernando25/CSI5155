import gc
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Optional
from skimage.color import rgb2gray
from tqdm import tqdm

class ClassifierSVM:
    """
    Improved classifier inspired by XRCE 2011: Dense RGB patches for color awareness + 
    grayscale patches for texture, PCA-reduced, GMM Fisher Vector encoding, normalized, 
    then linear SVM. Handles CIFAR-10-like small images robustly (no sparse keypoints).
    
    Args:
        n_patches: Approx. number of dense patches per image (grid-based).
        patch_size: Size of each patch (e.g., 8 for 32x32 images).
        n_components: GMM components for FV encoding.
        pca_dim: PCA output dim for descriptor compression.
        svm_C: SVM regularization.
        random_state: Random seed.
    """
    def __init__(
        self,
        n_patches: int = 16,  # e.g., 4x4 grid
        patch_size: int = 8,
        n_components: int = 32,
        pca_dim: int = 32,
        svm_C: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.n_components = n_components
        self.pca_dim = pca_dim
        self.svm_C = svm_C
        self.random_state = random_state
        self.pca = None
        self.gmm = None
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SGDClassifier(loss='hinge', alpha=svm_C, max_iter=1000, random_state=random_state))
        ])
    
    def _extract_dense_patches(self, images: List[np.ndarray], grayscale: bool = False) -> List[np.ndarray]:
        """Extract dense uniform patches from images (RGB or grayscale)."""
        descriptors_list = []
        h, w = 32, 32  # Assume CIFAR-10 size; resize if needed
        stride = max(1, int(np.sqrt(self.n_patches)))  # e.g., stride=4 for ~16 patches
        for img in tqdm(images, desc="Extracting patches", leave=False):
            if len(img.shape) == 3 and grayscale:
                img = rgb2gray(img)
            elif len(img.shape) == 2 and not grayscale:
                img = np.stack([img] * 3, axis=-1)  # Fake RGB if gray
            descs = []
            for i in range(0, h - self.patch_size + 1, stride):
                for j in range(0, w - self.patch_size + 1, stride):
                    patch = img[i:i+self.patch_size, j:j+self.patch_size]
                    if grayscale:
                        desc = patch.flatten()  # e.g., 64D for 8x8 gray
                    else:
                        desc = patch.flatten()  # 192D for 8x8x3 RGB
                    descs.append(desc)
            # Pad/truncate to fixed n_patches for consistency
            if len(descs) < self.n_patches:
                pad_desc = np.zeros_like(descs[0]) if descs else np.zeros(self.patch_size**2 * (1 if grayscale else 3))
                descs += [pad_desc] * (self.n_patches - len(descs))
            descriptors_list.append(np.array(descs[:self.n_patches]))
        return descriptors_list
    
    def _fit_features(self, X_train: List[np.ndarray], y_train: np.ndarray):
        """Fit PCA and GMM on training descriptors (called in fit)."""
        # Extract RGB (color) and gray (texture) descriptors
        rgb_descs_train = self._extract_dense_patches(X_train, grayscale=False)
        gray_descs_train = self._extract_dense_patches(X_train, grayscale=True)
        
        # Stack all for joint PCA (concat RGB + gray per patch for hybrid descs)
        all_descs = []
        for rgb_d, gray_d in tqdm(zip(rgb_descs_train, gray_descs_train), desc="Stacking descriptors", total=len(rgb_descs_train), leave=False):
            hybrid_desc = np.hstack([rgb_d, gray_d])  # e.g., 192 + 64 = 256D per patch
            all_descs.extend(hybrid_desc)
        all_descs = np.array(all_descs)
        
        # PCA compression
        self.pca = PCA(n_components=self.pca_dim, whiten=True, random_state=self.random_state)
        all_descs_pca = self.pca.fit_transform(all_descs)
        print(f"PCA explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        # GMM on PCA'd descriptors
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag', random_state=self.random_state)
        self.gmm.fit(all_descs_pca)
        print(f"Trained GMM with {self.n_components} components.")
    
    def _compute_fv(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute normalized Fisher Vector for a single image's descriptors."""
        
        # Apply PCA
        descriptors_pca = self.pca.transform(descriptors)
        
        # GMM posteriors and gradients
        N = len(descriptors_pca)
        if N == 0:
            return np.zeros(self.n_components + 2 * self.pca_dim * self.n_components)
        
        prob = self.gmm.predict_proba(descriptors_pca)
        weights = self.gmm.weights_
        means = self.gmm.means_
        covars = self.gmm.covariances_
        K, D = self.n_components, self.pca_dim
        
        d_pi = np.sum(prob, axis=0) / N - weights
        d_mu = np.zeros((K, D))
        d_sigma = np.zeros((K, D))
        
        for k in range(K):
            gamma_k = prob[:, k]
            sum_gamma = np.sum(gamma_k)
            if sum_gamma > 1e-6:
                diff = descriptors_pca - means[k]
                d_mu[k] = np.sum(gamma_k[:, np.newaxis] * diff, axis=0) / sum_gamma
                sigma_term = (diff ** 2 / covars[k] - 1)
                d_sigma[k] = -0.5 * np.sum(gamma_k[:, np.newaxis] * sigma_term, axis=0) / sum_gamma
        
        fv = np.hstack((d_pi, d_mu.ravel(), d_sigma.ravel()))
        
        # Normalize (power-law + L2)
        fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-8)
        fv /= (np.linalg.norm(fv) + 1e-8)
        return fv
    
    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract FV features for a list of images."""
        rgb_descs = self._extract_dense_patches(images, grayscale=False)
        gray_descs = self._extract_dense_patches(images, grayscale=True)
        
        fvs = []
        for rgb_d, gray_d in tqdm(zip(rgb_descs, gray_descs), desc="Computing Fisher Vectors", total=len(rgb_descs), leave=False):
            hybrid_desc = np.hstack([rgb_d, gray_d])  # Hybrid per patch
            fv = self._compute_fv(hybrid_desc)
            fvs.append(fv)
        return np.array(fvs)
    
    def fit(self, X: List[np.ndarray], y: np.ndarray) -> 'ClassifierSVM':
        X = np.array(X)
        self._fit_features(X, y)
        gc.collect()
        print("Extracting features...")
        X_features = self._extract_features(X)
        print("Fitting classifier...")
        self.classifier.fit(X_features, y)
        print("Classifier fitted.")
        return self
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        X_features = self._extract_features(X)
        return self.classifier.predict(X_features)
    
    def score(self, X: List[np.ndarray], y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)