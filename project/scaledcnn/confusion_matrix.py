from pathlib import Path

import torch

from device import device
from data import CIFAR10_CLASS_NAMES, get_cifar10_dataloader
from scaledcnn.eval import build_model_from_checkpoint
from utils import (
    collect_scaledcnn_predictions,
    generate_classification_report_and_confusion_matrix,
    require_file,
)


def run(
    model_path: str = ".cache/models/scaledcnn.pth",
    split: str = "test",
    batch_size: int = 128,
):
    model_path_obj = require_file(model_path, hint="Train the model first")

    checkpoint = torch.load(str(model_path_obj), map_location=device)
    model, config = build_model_from_checkpoint(checkpoint, device)

    k = config.get("k", 1)

    data_loader = get_cifar10_dataloader(
        split=split,
        batch_size=batch_size,
        shuffle=False,
    )
    predictions, labels = collect_scaledcnn_predictions(model, data_loader)

    class_names = CIFAR10_CLASS_NAMES
    model_token = f"scaledcnn_k{k}"

    result = generate_classification_report_and_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        model_token=model_token,
        split=split,
    )

    return Path(result["confusion_figure"])


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "confusion-matrix", help="Generate confusion matrix for ScaledCNN"
    )
    parser.add_argument("--model-path", default=".cache/models/scaledcnn.pth")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.set_defaults(
        entry=lambda args: run(
            model_path=args.model_path,
            split=args.split,
            batch_size=args.batch_size,
        )
    )
    return parser
