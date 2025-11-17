import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data import get_cifar10_class_names, get_cifar10_dataloader
from scaledcnn.evaluation import resolve_torch_device
from scaledcnn.model import ScaledCNN
from visualization import plot_confusion_matrix


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint.get(
        "config",
        {
            "k": 1,
            "num_classes": 10,
        },
    )
    model = ScaledCNN(
        k=config.get("k", 1),
        num_classes=config.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


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
    model_path: str = ".cache/models/scaledcnn.pth",
    split: str = "test",
    batch_size: int = 128,
    device: str | None = None,
    output_path: str | None = None,
):
    torch_device = resolve_torch_device(device)
    print(f"Using device: {torch_device}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train the model first: uv run main.py scaledcnn train")
        return None

    print(f"Loading model from: {model_path_obj}")
    checkpoint = torch.load(str(model_path_obj), map_location=torch_device)
    model, config = build_model_from_checkpoint(checkpoint, torch_device)
    print("Model loaded successfully!")

    k = config.get("k", 1)
    print(f"Model configuration: k={k}")

    print(f"\nLoading CIFAR-10 {split} dataset...")
    data_loader, torch_device = get_cifar10_dataloader(
        split=split,
        batch_size=batch_size,
        device=torch_device,
        shuffle=False,
    )
    print(f"{split.capitalize()} samples: {len(data_loader.dataset)}")

    print(f"\nComputing predictions on {split} set...")
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(torch_device)
            labels = labels.to(torch_device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)

    class_names = get_cifar10_class_names()
    model_name = f"ScaledCNN(k={k}) ({split})"
    model_token = f"scaledcnn_k{k}"
    if output_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / ".cache" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"scaledcnn_k{k}_{split}_confusion_matrix.pdf")

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
    _save_confusion_matrix(raw_cm, normalized_cm, class_names, model_token, split)

    return output


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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the figure. Default: auto-generated based on model",
    )

    def _entry(args):
        return run(
            model_path=args.model_path,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output_path,
        )

    parser.set_defaults(entry=_entry)
    return parser

