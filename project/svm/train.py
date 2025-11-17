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
from utils import require_file, load_pca, load_gmm


def _load_requirements():
    pca_path = require_file(PCA_PATH, hint="Train PCA first")
    fv_path = require_file(FISHER_VECTORS_PATH, hint="Compute fisher vectors first")
    gmm_path = require_file(GMM_PATH, hint="Compute fisher vectors first")
    labels_path = require_file(LABELS_PATH, hint="Extract patches first")

    return pca_path, fv_path, gmm_path, labels_path


def run(model_path: str = SVM_CLASSIFIER_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/svm/train/task_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir.resolve()}")

    pca_path, fv_path, gmm_path, labels_path = _load_requirements()

    start_time = time.time()

    pca = load_pca(pca_path)
    fvs = joblib.load(str(fv_path))
    y_train = joblib.load(str(labels_path))

    gmm = load_gmm(gmm_path)
    model = ClassifierSVM(
        pca=pca,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        n_components=N_COMPONENTS,
        svm_C=SVM_C,
        random_state=RANDOM_STATE,
    )

    model.gmm = gmm

    model.classifier.fit(fvs, y_train)
    train_accuracy = model.classifier.score(fvs, y_train)

    writer.add_scalar("train/accuracy", train_accuracy, 0)
    writer.add_text("config", f"C={SVM_C}", 0)

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_obj))

    elapsed = time.time() - start_time
    print(
        f"Training complete: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%) | Time: {elapsed:.2f}s"
    )
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
    parser.set_defaults(entry=lambda args: run(model_path=args.model_path))
    return parser
