import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    print("Loading CIFAR-10 dataset...")
    ds_dict = load_cifar10_data()
    
    print("Preparing training data...")
    X_train, y_train = prepare_data(ds_dict, "train")
    print(f"Training samples: {len(X_train)}")
    
    print("Preparing test data...")
    X_test, y_test = prepare_data(ds_dict, "test")
    print(f"Test samples: {len(X_test)}")
    
    # Check if cached features exist
    repo_root = Path(__file__).resolve().parents[1]
    feature_dir = repo_root / ".cache" / "features"
    feature_id = "n32_pca32_rgb+gray"
    
    train_features_path = feature_dir / f"X_train_{feature_id}.npy"
    train_labels_path = feature_dir / f"y_train_{feature_id}.npy"
    test_features_path = feature_dir / f"X_test_{feature_id}.npy"
    test_labels_path = feature_dir / f"y_test_{feature_id}.npy"
    
    use_cached = all(p.exists() for p in [train_features_path, train_labels_path, test_features_path, test_labels_path])
    
    if use_cached:
        print(f"\nLoading cached features from {feature_dir}...")
        X_train_features = np.load(train_features_path)
        y_train = np.load(train_labels_path)
        X_test_features = np.load(test_features_path)
        y_test = np.load(test_labels_path)
        print(f"Loaded cached features (train: {X_train_features.shape}, test: {X_test_features.shape})")
        
        print("\nInitializing classifier...")
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SGDClassifier(loss='hinge', alpha=1.0, max_iter=1000, random_state=42))
        ])
        
        print("Training classifier...")
        start_time = time.time()
        classifier.fit(X_train_features, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        print("\nEvaluating on test set...")
        start_time = time.time()
        test_accuracy = np.mean(classifier.predict(X_test_features) == y_test)
        test_time = time.time() - start_time
    else:
        print("\nCached features not found. Run: uv run python -m scripts.extract_features")
        print("Training model from scratch...")
        
        print("\nInitializing ClassifierSVM...")
        model = ClassifierSVM(
            n_patches=16,
            patch_size=8,
            n_components=32,
            pca_dim=32,
            svm_C=1.0,
            random_state=42
        )
        
        print("Training model...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        print("\nEvaluating on test set...")
        start_time = time.time()
        test_accuracy = model.score(X_test, y_test)
        test_time = time.time() - start_time
        print(f"Test evaluation completed in {test_time:.2f} seconds")
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Test evaluation time: {test_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()

