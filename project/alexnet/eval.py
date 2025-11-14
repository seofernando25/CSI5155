import torch

from alexnet.model import CIFARAlexNet
from common_net import run_checkpoint_evaluation_cli
from data import get_cifar10_class_names, get_cifar10_dataloader


def _build_model_from_checkpoint(
    checkpoint: dict, device: torch.device
) -> CIFARAlexNet:
    model = CIFARAlexNet(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run(
    model_path: str = ".cache/models/alexnet_cifar.pth",
    batch_size: int = 128,
    device: str | None = None,
):
    return run_checkpoint_evaluation_cli(
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        model_builder=_build_model_from_checkpoint,
        class_names=get_cifar10_class_names(),
        dataloader_factory=get_cifar10_dataloader,
        missing_checkpoint_hint="Please train the model first: uv run python -m alexnet.train",
    )


def add_subparser(subparsers):
    parser = subparsers.add_parser("eval", help="AlexNet eval")
    parser.add_argument("--model-path", default=".cache/models/alexnet_cifar.pth")
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
