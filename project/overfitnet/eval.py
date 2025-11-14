import torch

from common_net import run_checkpoint_evaluation_cli
from data import get_cifar10_class_names, get_cifar10_dataloader
from overfitnet.model import OverfitAlexNet


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint.get(
        "config",
        {
            "num_classes": 10,
        },
    )
    model = OverfitAlexNet(
        num_classes=config.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


def run(
    model_path: str = ".cache/models/overfitnet.pth",
    batch_size: int = 128,
    device: str | None = None,
):
    extracted_config: dict[str, object] | None = None

    def _model_builder(checkpoint: dict, device_obj: torch.device) -> OverfitAlexNet:
        nonlocal extracted_config
        model, config = build_model_from_checkpoint(checkpoint, device_obj)
        extracted_config = config
        return model

    def _on_checkpoint_loaded(_checkpoint: dict) -> None:
        if extracted_config is not None:
            print("Configuration: BatchNorm=enabled")

    class_names = get_cifar10_class_names()

    metrics = run_checkpoint_evaluation_cli(
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        model_builder=_model_builder,
        class_names=class_names,
        dataloader_factory=get_cifar10_dataloader,
        on_checkpoint_loaded=_on_checkpoint_loaded,
        missing_checkpoint_hint="Please train the model first: uv run python -m overfitnet.train",
    )

    metrics["config"] = extracted_config

    return metrics


def add_subparser(subparsers):
    parser = subparsers.add_parser("eval", help="OverfitNet eval")
    parser.add_argument("--model-path", default=".cache/models/overfitnet.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    def _entry(args):
        return run(
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
        )

    parser.set_defaults(entry=_entry)
    return parser
