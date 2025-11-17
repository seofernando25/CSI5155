from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data import get_cifar10_class_names, get_cifar10_split
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM
from utils import generate_classification_report_and_confusion_matrix, require_file


def run(
    model_path: str = SVM_CLASSIFIER_PATH,
    split: str = "test",
):
    model_path_obj = require_file(
        model_path,
        hint="Train the model first"
    )

    model = ClassifierSVM.load(str(model_path_obj))
    X, y = get_cifar10_split(split)
    predictions = model.predict(X)

    labels = np.array(y)
    class_names = get_cifar10_class_names()
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%) | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    # Generate confusion matrix, classification report, and save metrics
    result = generate_classification_report_and_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        model_token="SVM",
        split=split,
    )
    
    return result["classification_report"]


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "report", help="Generate metrics report and confusion matrix for SVM"
    )
    parser.add_argument("--model-path", default=SVM_CLASSIFIER_PATH)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.set_defaults(entry=lambda args: run(
        model_path=args.model_path,
        split=args.split,
    ))
    return parser


