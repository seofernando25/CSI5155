from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        dataset,
        transform=None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Lazy loading from dataset (preserves train/val/test splits and label noise from process.py)
        item = self.dataset[idx]
        image = np.asarray(item["img"], dtype=np.float32)
        label = int(item["label"])  # Label noise is already baked into train split

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


def load_cifar10_data():
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "processed_datasets" / "cifar10"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please run: uv run python -m data.download (and then uv run python -m data.process)"
        )

    return load_from_disk(str(dataset_path), keep_in_memory=True)


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


def get_cifar10_split(split: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Get CIFAR-10 split as list of images and labels for SVM evaluation."""
    ds_dict = load_cifar10_data()
    ds = ds_dict[split]
    
    images: List[np.ndarray] = []
    labels: List[int] = []
    for item in ds:
        image = np.asarray(item["img"], dtype=np.float32)
        # Keep as HWC format (not flattened) - model expects individual images
        images.append(image)
        labels.append(int(item["label"]))
    
    y = np.array(labels, dtype=np.int64)
    
    return images, y


def get_cifar10_dataloader(
    split: str,
    batch_size: int,
    device: torch.device | str | None = None,
    shuffle: bool = False,
    transform=None,
):
    ds_dict = load_cifar10_data()
    ds = ds_dict[split]

    if isinstance(device, str):
        torch_device = torch.device(device)
    elif device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = device

    dataset = CIFAR10Dataset(ds, transform=transform)

    from torch.utils.data import DataLoader

    is_cuda = torch_device.type == "cuda"
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
        prefetch_factor=4 if is_cuda else 2,
    )

    return data_loader, torch_device
