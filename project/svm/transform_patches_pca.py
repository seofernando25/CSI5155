from pathlib import Path

import numpy as np
import joblib
from tqdm import tqdm

from svm.constants import (
    PATCHES_PATH,
    PCA_PATH,
    PCA_PATCHES_PATH,
)


def main():
    # Check for required files
    patches_path = Path(PATCHES_PATH)
    if not patches_path.exists():
        print(f"ERROR: Patches file not found at {patches_path}")
        print("Please run: uv run python -m svm.extract_patches")
        return

    pca_path = Path(PCA_PATH)
    if not pca_path.exists():
        print(f"ERROR: PCA file not found at {pca_path}")
        print("Please run: uv run python -m svm.train_pca")
        return

    # Load patches
    print(f"Loading patches from: {patches_path}")
    rgb_descs = joblib.load(str(patches_path))
    print(f"Loaded {len(rgb_descs):,} images with patches")

    # Load PCA
    print(f"Loading pre-trained PCA from: {pca_path}")
    pca_data = joblib.load(str(pca_path))
    if isinstance(pca_data, dict) and "pca" in pca_data:
        pca = pca_data["pca"]
    elif hasattr(pca_data, "n_components"):  # Direct PCA object
        pca = pca_data
    else:
        print(f"ERROR: Invalid PCA file format: {pca_path}")
        return

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
