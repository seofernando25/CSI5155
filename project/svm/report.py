from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from data import get_cifar10_class_names, get_cifar10_split
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM


def _default_output_path(split: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_dir = repo_root / ".cache" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / f"svm_{split}_metrics.json"


def run(
    model_path: str = SVM_CLASSIFIER_PATH,
    split: str = "test",
    output_path: str | None = None,
):
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path_obj}. Please train the model first: "
            "uv run main.py svm train"
        )

    print(f"Loading model from: {model_path_obj}")
    model = ClassifierSVM.load(str(model_path_obj))
    print("Model loaded successfully!")

    print(f"\nLoading CIFAR-10 {split} dataset...")
    X, y = get_cifar10_split(split)
    print(f"{split.capitalize()} samples: {len(X)}")

    print("\nRunning inference...")
    predictions = model.predict(X)

    class_names = get_cifar10_class_names()
    acc = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro")
    weighted_f1 = f1_score(y, predictions, average="weighted")

    print("\nSummary:")
    print(f"  Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")

    metrics = classification_report(
        y,
        predictions,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    output_path_obj = Path(output_path) if output_path else _default_output_path(split)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with output_path_obj.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_path_obj}")
    return metrics


def add_subparser(subparsers):
    parser = subparsers.add_parser("report", help="Generate metrics report for SVM")
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
        help="Optional path to save the metrics JSON (default: .cache/metrics)",
    )

    def _entry(args):
        return run(
            model_path=args.model_path,
            split=args.split,
            output_path=args.output_path,
        )

    parser.set_defaults(entry=_entry)
    return parser


