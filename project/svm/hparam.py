from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data import load_cifar10_data, prepare_split
from svm.constants import (
    PCA_PATH,
    GMM_PATH,
    PATCH_SIZE,
    STRIDE,
    N_COMPONENTS,
    RANDOM_STATE,
)


def load_pretrained_components():
    pca_path = Path(PCA_PATH)
    if not pca_path.exists():
        print(f"ERROR: PCA file not found at {pca_path}")
        print("Please run: uv run python -m svm.train_pca")
        return None, None

    gmm_path = Path(GMM_PATH)
    if not gmm_path.exists():
        print(f"ERROR: GMM file not found at {gmm_path}")
        print("Please run: uv run python -m svm.compute_fisher_vectors")
        return None, None

    # Load PCA
    print(f"Loading PCA from: {pca_path}")
    pca_data = joblib.load(str(pca_path))
    if isinstance(pca_data, dict) and "pca" in pca_data:
        pca = pca_data["pca"]
    elif hasattr(pca_data, "n_components"):
        pca = pca_data
    else:
        print(f"ERROR: Invalid PCA file format: {pca_path}")
        return None, None

    # Load GMM
    print(f"Loading GMM from: {gmm_path}")
    gmm_data = joblib.load(str(gmm_path))

    # Extract sklearn GMM from dictionary
    if isinstance(gmm_data, dict):
        if "sklearn_gmm" in gmm_data:
            gmm = gmm_data["sklearn_gmm"]
        elif "torchgmm" in gmm_data:
            print(
                "ERROR: Old torchgmm format detected. Please re-run compute_fisher_vectors to regenerate with sklearn GMM."
            )
            return None, None
        else:
            print(f"ERROR: Invalid GMM file format: {gmm_path}")
            return None, None
    else:
        # Direct sklearn GMM object
        gmm = gmm_data

    return pca, gmm


def compute_fisher_vectors(images, pca, gmm):
    from skimage.feature import fisher_vector
    from tqdm import tqdm

    # Extract patches
    from svm.model import ClassifierSVM

    temp_model = ClassifierSVM(
        pca=pca,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        n_components=N_COMPONENTS,
        svm_C=1.0,
        random_state=RANDOM_STATE,
    )
    temp_model.gmm = gmm

    # Normalize images
    X = [img.astype(np.float32) / 255.0 for img in images]

    # Extract patches
    rgb_descs = temp_model._extract_patches(X)

    # Compute Fisher Vectors
    fvs = []
    for img_desc in tqdm(rgb_descs, desc="Computing Fisher Vectors", leave=False):
        desc_pca = pca.transform(img_desc)
        fv = fisher_vector(desc_pca, gmm, improved=True)
        fvs.append(fv)

    return np.array(fvs, dtype=np.float32)


def objective(trial, X_train_fv, y_train, X_val_fv, y_val):
    # Suggest C value from the specified list
    C = trial.suggest_categorical("C", [1e-5, 3e-5, 5e-5, 1e-4])

    # Create and train SVM classifier
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=C,
                    max_iter=100,
                    random_state=RANDOM_STATE,
                    dual=False,
                    tol=1e-3,  # Relaxed tolerance for faster training
                    verbose=0,  # Disable verbose output
                ),
            ),
        ]
    )

    # Train
    classifier.fit(X_train_fv, y_train)

    # Evaluate on validation set
    y_pred = classifier.predict(X_val_fv)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


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
    X_all, y_all = prepare_split(ds_dict, "train")
    print(f"Total training samples: {len(X_all)}")

    # Use 10% of training data for training, rest for validation
    n_train = int(len(X_all) * 0.75)
    indices = np.random.RandomState(RANDOM_STATE).permutation(len(X_all))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]  # Use remaining 90% for validation

    X_train = [X_all[i] for i in train_indices]
    y_train = y_all[train_indices]
    X_val = [X_all[i] for i in val_indices]
    y_val = y_all[val_indices]

    print(f"Train samples (75%): {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Load pre-trained components
    print("\nLoading pre-trained PCA and GMM...")
    pca, gmm = load_pretrained_components()
    if pca is None or gmm is None:
        return

    # Compute Fisher Vectors for training and validation sets
    print("\nComputing Fisher Vectors for training set...")
    X_train_fv = compute_fisher_vectors(X_train, pca, gmm)
    print(f"Training Fisher Vectors shape: {X_train_fv.shape}")

    print("\nComputing Fisher Vectors for validation set...")
    X_val_fv = compute_fisher_vectors(X_val, pca, gmm)
    print(f"Validation Fisher Vectors shape: {X_val_fv.shape}")

    # Create Optuna study
    print("\n" + "=" * 60)
    print("Starting hyperparameter optimization with Optuna...")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        study_name="svm_c_optimization",
        sampler=optuna.samplers.GridSampler({"C": [1e-5, 3e-5, 5e-5, 1e-4]}),
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train_fv, y_train, X_val_fv, y_val),
        n_trials=4,  # One trial per C value
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best C value: {study.best_params['C']}")
    print(
        f"Best validation accuracy: {study.best_value:.4f} ({study.best_value * 100:.2f}%)"
    )
    print("\nAll trials:")
    for trial in study.trials:
        C_val = trial.params["C"]
        print(
            f"  C={C_val:10.5f}: accuracy={trial.value:.4f} ({trial.value * 100:.2f}%)"
        )
    print("=" * 60)

    # Save results
    results_path = Path(".cache") / "svm_hparam_results.pkl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": study.trials,
        },
        str(results_path),
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
