import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dropout: float = 0.0,
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return layers


class CIFARAlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        conv_plan = [
            # Conv1: 32×32×3 -> 32×32×32
            {"out": 32, "kernel_size": 5, "padding": 2, "pool": True, "dropout": 0.0},
            # Conv2: 16×16×32 -> 16×16×64
            {"out": 64, "kernel_size": 5, "padding": 2, "pool": True, "dropout": 0.0},
            # Conv3: 8×8×64 -> 8×8×128
            {"out": 128, "kernel_size": 3, "padding": 1, "pool": False, "dropout": 0.1},
            # Conv4: 8×8×128 -> 8×8×128
            {"out": 128, "kernel_size": 3, "padding": 1, "pool": False, "dropout": 0.1},
            # Conv5: 8×8×128 -> 4×4×96 (after pooling)
            {"out": 96, "kernel_size": 3, "padding": 1, "pool": True, "dropout": 0.0},
        ]

        layers: list[nn.Module] = []
        in_channels = 3

        for spec in conv_plan:
            layers.extend(
                _conv_block(
                    in_channels,
                    spec["out"],
                    kernel_size=spec["kernel_size"],
                    padding=spec["padding"],
                    dropout=spec["dropout"],
                )
            )
            if spec["pool"]:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = spec["out"]

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(conv_plan[-1]["out"], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
