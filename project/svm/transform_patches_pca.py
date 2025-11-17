from pathlib import Path

import numpy as np
import joblib
from tqdm import tqdm

from svm.constants import (
    PATCHES_PATH,
    PCA_PATH,
    PCA_PATCHES_PATH,
)
from utils import require_file, load_pca


def main():
    # Check for required files
    patches_path = require_file(
        PATCHES_PATH,
        hint="Extract patches first"
    )
    pca_path = require_file(
        PCA_PATH,
        hint="Train PCA first"
    )


    # Load patches
    print(f"Loading patches from: {patches_path}")
    rgb_descs = joblib.load(str(patches_path))
    print(f"Loaded {len(rgb_descs):,} images with patches")

    # Load PCA
    print(f"Loading pre-trained PCA from: {pca_path}")
    pca = load_pca(pca_path)

    print(f"Loaded PCA with {pca.n_components} components")

    # Transform patches with PCA
    print("\nTransforming patches with PCA...")
    rgb_descs_pca = []
    for d in tqdm(rgb_descs, desc="PCA Transform"):
        desc_pca = pca.transform(d)
        rgb_descs_pca.append(desc_pca.astype(np.float32))

    all_patch_count = sum(len(d) for d in rgb_descs_pca)
    print(f"Transformed {all_patch_count:,} patches to {pca.n_components} dimensions")

    # Save PCA-transformed patches
    pca_patches_path = Path(PCA_PATCHES_PATH)
    pca_patches_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rgb_descs_pca, str(pca_patches_path))
    print(f"\nPCA-transformed patches saved to: {pca_patches_path}")

    print("\nSummary:")
    print(f"  - Images: {len(rgb_descs_pca):,}")
    print(f"  - Total patches: {all_patch_count:,}")
    print(f"  - PCA dimension: {pca.n_components}")


if __name__ == "__main__":
    main()
