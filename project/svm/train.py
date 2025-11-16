from pathlib import Path

from datetime import datetime
import time
import joblib
from torch.utils.tensorboard import SummaryWriter

from svm.model import ClassifierSVM
from svm.constants import (
    SVM_CLASSIFIER_PATH,
    PCA_PATH,
    FISHER_VECTORS_PATH,
    GMM_PATH,
    LABELS_PATH,
    PATCH_SIZE,
    STRIDE,
    N_COMPONENTS,
    SVM_C,
    RANDOM_STATE,
)


def _load_requirements():
    pca_path = Path(PCA_PATH)
    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA file not found at {pca_path}. Please run: uv run python -m svm.train_pca"
        )

    fv_path = Path(FISHER_VECTORS_PATH)
    if not fv_path.exists():
        raise FileNotFoundError(
            f"Fisher Vectors file not found at {fv_path}. Please run: uv run python -m svm.compute_fisher_vectors"
        )

    gmm_path = Path(GMM_PATH)
    if not gmm_path.exists():
        raise FileNotFoundError(
            f"GMM file not found at {gmm_path}. Please run: uv run python -m svm.compute_fisher_vectors"
        )

    labels_path = Path(LABELS_PATH)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_path}. Please run: uv run python -m svm.extract_patches"
        )

    return pca_path, fv_path, gmm_path, labels_path


def run(model_path: str = SVM_CLASSIFIER_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/svm/train/task_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir.resolve()}")

    pca_path, fv_path, gmm_path, labels_path = _load_requirements()

    start_time = time.time()

    print(f"Loading pre-trained PCA from: {pca_path}")
    pca_data = joblib.load(str(pca_path))
    if isinstance(pca_data, dict) and "pca" in pca_data:
        pca = pca_data["pca"]
    elif hasattr(pca_data, "n_components"):
        pca = pca_data
    else:
        raise ValueError(f"Invalid PCA file format: {pca_path}")
    print(f"Loaded PCA with {pca.n_components} components")

    print(f"\nLoading cached Fisher Vectors from: {fv_path}")
    fvs = joblib.load(str(fv_path))
    print(f"Loaded Fisher Vectors: {fvs.shape}")

    print(f"Loading cached labels from: {labels_path}")
    y_train = joblib.load(str(labels_path))
    print(f"Loaded labels: {len(y_train)} samples")

    if len(fvs) != len(y_train):
        raise ValueError(
            f"Mismatch - Fisher Vectors has {len(fvs)} samples, but labels has {len(y_train)} samples"
        )

    print(f"Loading cached GMM from: {gmm_path}")
    gmm_data = joblib.load(str(gmm_path))

    if isinstance(gmm_data, dict):
        if "sklearn_gmm" in gmm_data:
            gmm = gmm_data["sklearn_gmm"]
        else:
            raise ValueError(
                f"Invalid GMM file format. Expected 'sklearn_gmm' key, got: {list(gmm_data.keys())}"
            )
    else:
        gmm = gmm_data

    print("\nInitializing ClassifierSVM...")
    model = ClassifierSVM(
        pca=pca,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        n_components=N_COMPONENTS,
        svm_C=SVM_C,
        random_state=RANDOM_STATE,
    )

    model.gmm = gmm

    print("\nTraining SVM on cached Fisher Vectors...")
    model.classifier.fit(fvs, y_train)
    train_accuracy = model.classifier.score(fvs, y_train)
    print(f"SVM classifier fitted with {len(fvs)} samples")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")

    writer.add_scalar("train/accuracy", train_accuracy, 0)
    writer.add_text("config", f"C={SVM_C}", 0)

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_obj))
    print(f"\nModel saved to: {model_path_obj}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f} seconds")
    writer.add_scalar("train/time_seconds", elapsed, 0)

    hparams = {
        "svm_C": SVM_C,
        "n_components": N_COMPONENTS,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "random_state": RANDOM_STATE,
    }
    metrics = {
        "hparam/train_accuracy": train_accuracy,
        "hparam/time_seconds": elapsed,
    }
    writer.add_hparams(hparams, metrics)

    writer.close()
    return {
        "accuracy": train_accuracy,
        "model_path": model_path_obj,
        "log_dir": log_dir,
        "time_seconds": elapsed,
    }


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", help="Train SVM classifier")
    parser.add_argument("--model-path", default=SVM_CLASSIFIER_PATH)

    def _entry(args):
        return run(model_path=args.model_path)

    parser.set_defaults(entry=_entry)
    return parser
