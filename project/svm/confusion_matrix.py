import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from data import get_cifar10_class_names, get_cifar10_split
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM
from visualization import plot_confusion_matrix


def _save_confusion_matrix(
    matrix: np.ndarray,
    normalized_matrix: np.ndarray,
    class_names: list[str],
    model_token: str,
    split: str,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_dir = repo_root / ".cache" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"{model_token}_{split}_confusion_matrix.json"
    payload = {
        "model": model_token,
        "split": split,
        "class_names": class_names,
        "matrix": matrix.tolist(),
        "normalized_matrix": normalized_matrix.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Confusion matrix counts saved to: {output_path}")
    return output_path


def run(
    model_path: str = SVM_CLASSIFIER_PATH,
    split: str = "test",
    output_path: str | None = None,
):
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train the model first: uv run main.py svm train")
        return None

    print(f"Loading model from: {model_path_obj}")
    model = ClassifierSVM.load(str(model_path_obj))
    print("Model loaded successfully!")

    print(f"\nLoading CIFAR-10 {split} dataset...")
    try:
        X, y = get_cifar10_split(split)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return None
    print(f"{split.capitalize()} samples: {len(X)}")

    print(f"\nComputing predictions on {split} set...")
    predictions = model.predict(X)

    labels = np.array(y)
    class_names = get_cifar10_class_names()
    model_name = f"SVM ({split})"
    output = plot_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        model_name=model_name,
        output_path=output_path,
    )
    raw_cm = confusion_matrix(labels, predictions)
    row_sums = raw_cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    normalized_cm = raw_cm.astype(np.float64) / row_sums
    _save_confusion_matrix(raw_cm, normalized_cm, class_names, "svm", split)

    return output


def add_subparser(subparsers):
    parser = subparsers.add_parser("confusion-matrix", help="Generate confusion matrix for SVM")
    parser.add_argument("--model-path", default=SVM_CLASSIFIER_PATH)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the figure. Default: .cache/figures/svm_confusion_matrix.pdf",
    )

    def _entry(args):
        return run(
            model_path=args.model_path,
            split=args.split,
            output_path=args.output_path,
        )

    parser.set_defaults(entry=_entry)
    return parser

