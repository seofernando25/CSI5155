import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from models.classifier_svm import ClassifierSVM


def load_cifar10_data():
    """Load base CIFAR-10 dataset (RGB images)."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "base_datasets" / "cifar10"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Base dataset not found at {dataset_path}. "
            "Please run: uv run python -m scripts.data_download"
        )
    
    ds_dict = load_from_disk(str(dataset_path))
    return ds_dict


def prepare_data(ds_dict, split: str, max_samples: int = None):
    """Extract images and labels from dataset split."""
    ds = ds_dict[split]
    
    # Convert images to numpy arrays
    images = [np.asarray(img) for img in ds["img"]]
    labels = np.array(ds["label"])
    
    if max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]
    
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Extract and cache Fisher Vector features")
    parser.add_argument("--n-components", type=int, default=32, help="GMM components (default: 32)")
    parser.add_argument("--pca-dim", type=int, default=32, help="PCA dimension (default: 32)")
    parser.add_argument("--output-dir", type=str, default=".cache/features", help="Output directory for cached features")
    args = parser.parse_args()
    
    print("Loading CIFAR-10 dataset...")
    ds_dict = load_cifar10_data()
    
    print("Preparing training data...")
    X_train, y_train = prepare_data(ds_dict, "train")
    print(f"Training samples: {len(X_train)}")
    
    print("Preparing test data...")
    X_test, y_test = prepare_data(ds_dict, "test")
    print(f"Test samples: {len(X_test)}")
    
    # Create output directory
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create feature identifier string (always rgb+gray)
    feature_id = f"n{args.n_components}_pca{args.pca_dim}_rgb+gray"
    
    print(f"\nInitializing ClassifierSVM (features: {feature_id})...")
    model = ClassifierSVM(
        n_patches=16,
        patch_size=8,
        n_components=args.n_components,
        pca_dim=args.pca_dim,
        svm_C=1.0,
        random_state=42
    )
    
    print("Fitting PCA and GMM on training data...")
    model._fit_features(X_train, y_train)
    
    print("Extracting training features...")
    X_train_features = model._extract_features(X_train)
    
    print("Extracting test features...")
    X_test_features = model._extract_features(X_test)
    
    # Save features and labels
    train_features_path = output_dir / f"X_train_{feature_id}.npy"
    train_labels_path = output_dir / f"y_train_{feature_id}.npy"
    test_features_path = output_dir / f"X_test_{feature_id}.npy"
    test_labels_path = output_dir / f"y_test_{feature_id}.npy"
    
    print(f"\nSaving features to {output_dir}...")
    np.save(train_features_path, X_train_features)
    np.save(train_labels_path, y_train)
    np.save(test_features_path, X_test_features)
    np.save(test_labels_path, y_test)
    
    print(f"Saved:")
    print(f"  Training features: {train_features_path} (shape: {X_train_features.shape})")
    print(f"  Training labels: {train_labels_path} (shape: {y_train.shape})")
    print(f"  Test features: {test_features_path} (shape: {X_test_features.shape})")
    print(f"  Test labels: {test_labels_path} (shape: {y_test.shape})")
    print(f"\nFeature ID: {feature_id}")
    print("Done!")


if __name__ == "__main__":
    main()

