from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data.datasets import load_cifar10_data
from svm.constants import (
    PCA_PATH,
    GMM_PATH,
    PATCH_SIZE,
    STRIDE,
    N_COMPONENTS,
    RANDOM_STATE,
)
from utils.paths import require_file
from svm.loaders import load_pca, load_gmm


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

    X = [np.asarray(img, dtype=np.float32) for img in images]
    rgb_descs = temp_model._extract_patches(X)
    fvs = []
    for img_desc in tqdm(rgb_descs, desc="Computing Fisher Vectors", leave=False):
        desc_pca = pca.transform(img_desc)
        fv = fisher_vector(desc_pca, gmm, improved=True)
        fvs.append(fv)

    return np.array(fvs, dtype=np.float32)


def objective(trial, X_train_fv, y_train, X_val_fv, y_val):
    # These values have been fine tuned from previously trying larger values already
    # eg: [0.01, 0.1, 1, 10] but these didn't perform well at all compared to smaller values
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
                    tol=1e-3,
                    verbose=0,
                ),
            ),
        ]
    )

    classifier.fit(X_train_fv, y_train)
    y_pred = classifier.predict(X_val_fv)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


def main():
    ds_dict = load_cifar10_data()

    # Prepare training and validation data
    train_ds = ds_dict["train"]
    X_train = [np.asarray(item["img"], dtype=np.float32) for item in train_ds]
    y_train = np.array([item["label"] for item in train_ds])

    val_ds = ds_dict["validation"]
    X_val = [np.asarray(item["img"], dtype=np.float32) for item in val_ds]
    y_val = np.array([item["label"] for item in val_ds])

    # Load pre-trained components
    pca = load_pca(require_file(PCA_PATH, hint="Train PCA first"))
    gmm = load_gmm(require_file(GMM_PATH, hint="Train GMM first"))

    # Compute Fisher Vectors for training and validation sets
    X_train_fv = compute_fisher_vectors(X_train, pca, gmm)
    X_val_fv = compute_fisher_vectors(X_val, pca, gmm)

    study = optuna.create_study(
        direction="maximize",
        study_name="svm_c_optimization",
        sampler=optuna.samplers.GridSampler({"C": [1e-5, 3e-5, 5e-5, 1e-4]}),
    )

    study.optimize(
        lambda trial: objective(trial, X_train_fv, y_train, X_val_fv, y_val),
        n_trials=4,
        show_progress_bar=True,
    )

    print(
        f"\nBest: {study.best_value:.4f} ({study.best_value * 100:.2f}%) | C: {study.best_params['C']}"
    )

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


if __name__ == "__main__":
    main()
