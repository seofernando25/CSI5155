from typing import List

import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return layers


class OverfitAlexNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        conv_plan = [
            # Conv1: 32×32×3 -> 32×32×32
            {"out": 32, "kernel_size": 5, "padding": 2, "pool": True},
            # Conv2: 16×16×32 -> 16×16×64
            {"out": 64, "kernel_size": 5, "padding": 2, "pool": True},
            # Conv3: 8×8×64 -> 8×8×128
            {"out": 128, "kernel_size": 3, "padding": 1, "pool": False},
            # Conv4: 8×8×128 -> 8×8×128
            {"out": 128, "kernel_size": 3, "padding": 1, "pool": False},
            # Conv5: 8×8×128 -> 4×4×256 (after pooling)
            {"out": 256, "kernel_size": 3, "padding": 1, "pool": True},
            # Conv6: 4×4×256 -> 4×4×256
            {"out": 256, "kernel_size": 3, "padding": 1, "pool": False},
            # Conv7: 4×4×256 -> 4×4×128
            {"out": 128, "kernel_size": 3, "padding": 1, "pool": False},
        ]

        layers: List[nn.Module] = []
        in_channels = 3

        for spec in conv_plan:
            layers.extend(
                _conv_block(
                    in_channels,
                    spec["out"],
                    kernel_size=spec["kernel_size"],
                    padding=spec["padding"],
                )
            )
            if spec["pool"]:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = spec["out"]

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_plan[-1]["out"], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
