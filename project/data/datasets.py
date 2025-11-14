from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        images: Iterable[np.ndarray],
        labels: Sequence[int],
        transform=None,
    ) -> None:
        self.images = list(images)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            image = self.transform(image)
        else:
            # Images are already normalized to 0-1 range from processing
            # Just convert to tensor and permute channels
            if image.dtype == np.uint8:
                # Fallback: if somehow uint8, normalize
                image = image.astype(np.float32) / 255.0
            else:
                # Already float32 in 0-1 range
                image = image.astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label


def _resolve_dataset_path(dataset_root: Optional[Path]) -> Path:
    if dataset_root is not None:
        return Path(dataset_root)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".cache" / "processed_datasets" / "cifar10"


def load_cifar10_data(dataset_root: Optional[Path] = None):
    dataset_path = _resolve_dataset_path(dataset_root)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please run: uv run python -m data.download (and then uv run python -m data.process)"
        )

    return load_from_disk(str(dataset_path))


def prepare_split(ds_dict, split: str):
    ds = ds_dict[split]
    images = [np.asarray(img) for img in ds["img"]]
    labels = np.array(ds["label"])
    return images, labels


CIFAR10_CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_class_names() -> List[str]:
    return CIFAR10_CLASS_NAMES.copy()


def get_cifar10_split(split: str, dataset_root: Optional[Path] = None):
    ds_dict = load_cifar10_data(dataset_root)
    return prepare_split(ds_dict, split)


def get_cifar10_dataloader(
    split: str,
    batch_size: int,
    device: torch.device | str | None = None,
    shuffle: bool = False,
    transform=None,
    dataset_root: Optional[Path] = None,
):
    images, labels = get_cifar10_split(split, dataset_root)

    if isinstance(device, str):
        torch_device = torch.device(device)
    elif device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = device

    dataset = CIFAR10Dataset(images, labels, transform=transform)

    from common_net.training import build_dataloader

    data_loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        device=torch_device,
        shuffle=shuffle,
    )

    return data_loader, torch_device
