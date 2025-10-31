import argparse
from model import ShaderMLP


def count_params(model: ShaderMLP) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters of ShaderMLP")
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_sine_layers', type=int, default=10)
    parser.add_argument('--first_omega_0', type=float, default=120.0)
    parser.add_argument('--hidden_omega_0', type=float, default=60.0)
    args = parser.parse_args()

    model = ShaderMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_sine_layers=args.num_sine_layers,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
        spectral_centroid=None,
    )

    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")


if __name__ == '__main__':
    main()



