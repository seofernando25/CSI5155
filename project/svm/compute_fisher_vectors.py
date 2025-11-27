from pathlib import Path

import numpy as np
import joblib
from skimage.feature import fisher_vector
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from svm.constants import (
    PCA_PATCHES_PATH,
    PCA_PATH,
    FISHER_VECTORS_PATH,
    GMM_PATH,
    N_COMPONENTS,
    RANDOM_STATE,
)
from svm.loaders import load_pca


def main():
    # Load PCA-transformed patches
    print(f"Loading PCA-transformed patches from: {PCA_PATCHES_PATH}")
    rgb_descs_pca = joblib.load(PCA_PATCHES_PATH)
    print(f"Loaded {len(rgb_descs_pca):,} images with PCA-transformed patches")

    # Load PCA to get dimension
    print(f"Loading PCA from: {PCA_PATH}")
    pca = load_pca(PCA_PATH)
    pca_dim = int(pca.n_components)
    print(f"PCA dimension: {pca_dim}")

    np.random.seed(RANDOM_STATE)

    # Fit GMM on all patches
    print(f"\nFitting GMM with {N_COMPONENTS} components...")
    all_descs_pca = np.concatenate([np.asarray(arr) for arr in rgb_descs_pca], axis=0)
    print(f"Total patches for GMM fitting: {len(all_descs_pca):,}")

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        max_iter=500,
        tol=1e-3,
        reg_covar=1e-6,
        init_params="kmeans",
        n_init=1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    print("Fitting GMM...")
    gmm.fit(all_descs_pca)
    print(f"GMM fitted with {N_COMPONENTS} components")

    # Compute GMM goodness metrics
    print("\nComputing GMM goodness metrics...")
    log_likelihood = gmm.score_samples(all_descs_pca).sum()
    aic = gmm.aic(all_descs_pca)
    bic = gmm.bic(all_descs_pca)
    n_params = N_COMPONENTS * (2 * pca_dim + 1) - 1

    print("\nGMM Goodness Metrics:")
    print(f"  - Log-likelihood: {log_likelihood:.2f}")
    print(f"  - AIC (Akaike Information Criterion): {aic:.2f} (lower is better)")
    print(f"  - BIC (Bayesian Information Criterion): {bic:.2f} (lower is better)")
    print(f"  - Number of samples: {len(all_descs_pca):,}")
    print(f"  - Number of parameters: {n_params:,}")

    # Save GMM
    Path(GMM_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"sklearn_gmm": gmm}, GMM_PATH)
    print(f"GMM saved to: {GMM_PATH}")

    # Compute Fisher Vectors
    print("\nComputing Fisher Vectors...")
    fvs = []
    for desc_pca in tqdm(rgb_descs_pca, desc="FV compute"):
        fv = fisher_vector(np.asarray(desc_pca), gmm, improved=True)
        fvs.append(fv)
    fvs = np.array(fvs, dtype=np.float32)

    print(f"Computed Fisher Vectors: {fvs.shape}")

    # Save Fisher Vectors
    Path(FISHER_VECTORS_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fvs, FISHER_VECTORS_PATH)
    print(f"\nFisher Vectors saved to: {FISHER_VECTORS_PATH}")

    print("\nSummary:")
    print(f"  - Images: {len(fvs):,}")
    print(f"  - Fisher Vector dimension: {fvs.shape[1]:,}")
    print(f"  - GMM components: {N_COMPONENTS}")
    print(f"  - PCA dimension: {pca_dim}")
    print(f"  - GMM Log-likelihood: {log_likelihood:.2f}")
    print(f"  - GMM AIC: {aic:.2f}")
    print(f"  - GMM BIC: {bic:.2f}")


if __name__ == "__main__":
    main()
