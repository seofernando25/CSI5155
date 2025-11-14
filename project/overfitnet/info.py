from __future__ import annotations

from overfitnet.model import OverfitAlexNet


def describe_model() -> dict[str, int]:
    model = OverfitAlexNet(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("OverfitNet architecture:\n")
    print(model)
    print("\nParameter summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def run() -> dict[str, int]:
    return describe_model()


def add_subparser(subparsers):
    parser = subparsers.add_parser("info", help="Show OverfitNet model details")

    def _entry(_args):
        return run()

    parser.set_defaults(entry=_entry)
    return parser




