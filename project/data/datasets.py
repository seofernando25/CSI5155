from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.paths import require_file
from device import device
from torch.utils.data import DataLoader


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        dataset,
    ) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = np.asarray(self.dataset[idx]["img"], dtype=np.float32)
        # (Height, Width, Channels) -> (Channels, Height, Width)
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = int(self.dataset[idx]["label"])

        return image, label


class NumpyDatasetSplit:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels
        assert len(images) == len(labels), "Images and labels must have same length"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"img": self.images[idx], "label": int(self.labels[idx])}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class NumpyDatasetDict:
    def __init__(self, splits: Dict[str, NumpyDatasetSplit]):
        self.splits = splits

    def __getitem__(self, key: str) -> NumpyDatasetSplit:
        if key not in self.splits:
            raise KeyError(
                f"Split '{key}' not found. Available splits: {list(self.splits.keys())}"
            )
        return self.splits[key]

    def keys(self):
        return self.splits.keys()

    def items(self):
        return self.splits.items()


def load_cifar10_data():
    """Load CIFAR-10 dataset from processed numpy arrays."""

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = require_file(
        repo_root / ".cache" / "processed_datasets" / "cifar10",
        hint="Download and process the dataset first",
    )

    # Load numpy arrays for each split
    splits = {}
    split_names = ["train", "validation", "test"]

    for split_name in split_names:
        images_path = dataset_path / f"{split_name}_images.npy"
        labels_path = dataset_path / f"{split_name}_labels.npy"

        require_file(images_path, hint="Process the dataset first")
        require_file(labels_path, hint="Process the dataset first")

        images = np.load(str(images_path))
        labels = np.load(str(labels_path))

        # Ensure images are float32 in [0, 1] range (as processed by process.py)
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0

        splits[split_name] = NumpyDatasetSplit(images, labels)

    return NumpyDatasetDict(splits)


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


def get_cifar10_split(split: str) -> Tuple[List[np.ndarray], np.ndarray]:
    ds_dict = load_cifar10_data()
    ds = ds_dict[split]
    images = [ds.images[i] for i in range(len(ds))]
    y = ds.labels.astype(np.int64)

    return images, y


def get_cifar10_dataloader(
    split: str,
    batch_size: int,
    shuffle: bool = False,
):
    ds_dict = load_cifar10_data()
    ds = ds_dict[split]

    dataset = CIFAR10Dataset(ds)

    is_cuda = device.type == "cuda"
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
        prefetch_factor=4 if is_cuda else 2,
    )

    return data_loader
