from pathlib import Path

import torch

from scaledcnn.evaluation import run_checkpoint_evaluation_cli
from data.datasets import CIFAR10_CLASS_NAMES, get_cifar10_dataloader
from scaledcnn.model import ScaledCNN


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


def run(
    k: int,
    batch_size: int = 128,
):
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / ".cache" / "models" / f"scaledcnn_k{k}.pth"

    extracted_config: dict[str, object] | None = None

    def _model_builder(checkpoint: dict, device_obj: torch.device) -> ScaledCNN:
        nonlocal extracted_config
        model, config = build_model_from_checkpoint(checkpoint, device_obj)
        extracted_config = config
        return model

    def _on_checkpoint_loaded(_checkpoint: dict) -> None:
        if extracted_config is not None:
            k = extracted_config.get("k", 1)
            print(f"Configuration: k={k}")

    class_names = CIFAR10_CLASS_NAMES

    metrics = run_checkpoint_evaluation_cli(
        model_path=str(model_path),
        batch_size=batch_size,
        model_builder=_model_builder,
        class_names=class_names,
        dataloader_factory=get_cifar10_dataloader,
        on_checkpoint_loaded=_on_checkpoint_loaded,
        missing_checkpoint_hint="Train the model first",
    )

    metrics["config"] = extracted_config

    return metrics


def add_subparser(subparsers):
    parser = subparsers.add_parser("eval", help="ScaledCNN eval")
    parser.add_argument("--k", type=int, required=True, help="Scaling factor k")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.set_defaults(entry=lambda args: run(k=args.k, batch_size=args.batch_size))
    return parser
