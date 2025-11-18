from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import joblib
import gc

from svm.constants import (
    PCA_PATH,
    PATCHES_PATH,
    PATCH_SIZE,
    PCA_DIM,
)
from utils.paths import require_file


def main():
    # Check for required files
    patches_path = require_file(PATCHES_PATH, hint="Extract patches first")

    # Load patches
    print(f"Loading patches from: {patches_path}")
    rgb_descs = joblib.load(str(patches_path))
    all_patch_count = sum(len(d) for d in rgb_descs)
    print(f"Loaded {len(rgb_descs):,} images with {all_patch_count:,} total patches")

    # Fit PCA in batches to avoid OOM
    print(f"\nFitting PCA with {PCA_DIM} components...")
    pca = IncrementalPCA(n_components=PCA_DIM, whiten=True)

    batch_size = min(40000, len(rgb_descs) * 10)  # Adjust based on your RAM
    print(f"Using batch size: {batch_size:,} patches per batch")

    current_batch = []
    current_batch_size = 0

    for img_patches in tqdm(rgb_descs, desc="PCA batch fit"):
        if current_batch_size + len(img_patches) <= batch_size:
            current_batch.append(img_patches)
            current_batch_size += len(img_patches)
        else:
            # Fit on current batch
            if current_batch:
                batch = np.concatenate(current_batch, axis=0)
                pca.partial_fit(batch)
            # Start new batch
            current_batch = [img_patches]
            current_batch_size = len(img_patches)

    # Fit on remaining batch
    if current_batch:
        batch = np.concatenate(current_batch, axis=0)
        pca.partial_fit(batch)

    gc.collect()

    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(
        f"PCA explained variance: {explained_variance:.4f} ({explained_variance * 100:.2f}%)"
    )

    # Save PCA
    pca_path = Path(PCA_PATH)
    pca_path.parent.mkdir(parents=True, exist_ok=True)

    pca_data = {
        "pca": pca,
        "hyperparameters": {
            "patch_size": PATCH_SIZE,
            "pca_dim": PCA_DIM,
        },
    }

    joblib.dump(pca_data, str(pca_path))
    print(f"\nPCA saved to: {pca_path}")
    print(f"  - Patch size: {PATCH_SIZE}")
    print(f"  - PCA dimension: {PCA_DIM}")
    print(f"  - Explained variance: {explained_variance:.4f}")


if __name__ == "__main__":
    main()
