from typing import List

import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return layers


class ScaledCNN(nn.Module):
    def __init__(
        self,
        k: int = 1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "k", k)

        layers: List[nn.Module] = []
        in_channels = 3

        # Block 1: Conv(3, 16k, 3, padding=1) → BatchNorm → ReLU → MaxPool
        layers.extend(_conv_block(in_channels, 16 * k, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = 16 * k

        # Block 2: Conv(16k, 32k, 3, padding=1) → BatchNorm → ReLU → MaxPool
        layers.extend(_conv_block(in_channels, 32 * k, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = 32 * k

        # Block 3: Conv(32k, 64k, 3, padding=1) → BatchNorm → ReLU → AdaptiveAvgPool
        layers.extend(_conv_block(in_channels, 64 * k, kernel_size=3, padding=1))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*layers)

        # Flatten → Linear(64k, 10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * k, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
