from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from data import load_cifar10_data
from svm.constants import (
    PATCH_SIZE,
    STRIDE,
    PATCHES_PATH,
    LABELS_PATH,
)


def extract_patches_2d_stride(image, patch_size, stride):
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


def extract_patches(images, patch_size: int, stride: int):
    descriptors_list = []

    for img in tqdm(images, desc="Extracting patches", leave=False):
        patches = extract_patches_2d_stride(img, patch_size, stride)
        # Reshape and convert to float32 for memory efficiency
        descriptors_list.append(patches.reshape(len(patches), -1).astype(np.float32))

    return descriptors_list


def main():
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    try:
        ds_dict = load_cifar10_data()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    # Prepare training data
    print("Preparing training data...")
    train_ds = ds_dict["train"]
    # Images from processed dataset are already float32 in [0,1]
    X_train = [np.asarray(item["img"], dtype=np.float32) for item in train_ds]
    y_train = np.array([item["label"] for item in train_ds])
    print(f"Training samples: {len(X_train)}")

    # Extract patches
    print(f"\nExtracting patches (patch_size={PATCH_SIZE}, stride={STRIDE})...")
    rgb_descs = extract_patches(X_train, PATCH_SIZE, STRIDE)
    all_patch_count = sum(len(d) for d in rgb_descs)
    print(f"Total patches extracted: {all_patch_count:,}")

    # Save patches and labels
    patches_path = Path(PATCHES_PATH)
    patches_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rgb_descs, str(patches_path))
    print(f"\nPatches saved to: {patches_path}")

    labels_path = Path(LABELS_PATH)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(y_train, str(labels_path))
    print(f"Labels saved to: {labels_path}")

    print("\nSummary:")
    print(f"  - Images: {len(X_train):,}")
    print(f"  - Patches per image: {len(rgb_descs[0]):,}")
    print(f"  - Total patches: {all_patch_count:,}")
    print(f"  - Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(
        f"  - Stride: {STRIDE} ({'non-overlapping' if STRIDE == PATCH_SIZE else 'overlapping'})"
    )
    print(f"  - Patch dimension: {rgb_descs[0].shape[1]}")


if __name__ == "__main__":
    main()
