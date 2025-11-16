from __future__ import annotations

from scaledcnn.model import ScaledCNN


def describe_model(k: int = 1) -> dict[str, int]:
    model = ScaledCNN(k=k, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"ScaledCNN(k={k}) architecture:\n")
    print(model)
    print("\nParameter summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return {
        "k": k,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def run(k: int = 1) -> dict[str, int]:
    return describe_model(k=k)


def add_subparser(subparsers):
    parser = subparsers.add_parser("info", help="Show ScaledCNN model details")
    parser.add_argument("-k", "--k", type=int, default=1, help="Scaling factor k")

    def _entry(args):
        return run(k=args.k)

    parser.set_defaults(entry=_entry)
    return parser
