from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
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


class NumpyDatasetSplit:
    """Wrapper for numpy array dataset split that mimics HuggingFace Dataset interface."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels
        assert len(images) == len(labels), "Images and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return item in format expected by existing code: {'img': array, 'label': int}"""
        return {
            "img": self.images[idx],
            "label": int(self.labels[idx])
        }
    
    def __iter__(self):
        """Allow iteration over the dataset."""
        for i in range(len(self)):
            yield self[i]


class NumpyDatasetDict:
    """Wrapper for numpy array dataset that mimics HuggingFace DatasetDict interface."""
    
    def __init__(self, splits: Dict[str, NumpyDatasetSplit]):
        self.splits = splits
    
    def __getitem__(self, key: str) -> NumpyDatasetSplit:
        if key not in self.splits:
            raise KeyError(f"Split '{key}' not found. Available splits: {list(self.splits.keys())}")
        return self.splits[key]
    
    def keys(self):
        return self.splits.keys()
    
    def items(self):
        return self.splits.items()


def load_cifar10_data():
    """Load CIFAR-10 dataset from processed numpy arrays."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "processed_datasets" / "cifar10"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please run: uv run python -m data.download (and then uv run python -m data.process)"
        )

    # Load numpy arrays for each split
    splits = {}
    split_names = ["train", "validation", "test"]
    
    for split_name in split_names:
        images_path = dataset_path / f"{split_name}_images.npy"
        labels_path = dataset_path / f"{split_name}_labels.npy"
        
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing files for split '{split_name}': "
                f"expected {images_path} and {labels_path}. "
                "Please run: uv run python -m data.process"
            )
        
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


def get_cifar10_class_names() -> List[str]:
    return CIFAR10_CLASS_NAMES.copy()


def get_cifar10_split(split: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Get CIFAR-10 split as list of images and labels for SVM evaluation."""
    ds_dict = load_cifar10_data()
    ds = ds_dict[split]
    
    # Directly access numpy arrays from NumpyDatasetSplit
    if isinstance(ds, NumpyDatasetSplit):
        # Convert to list of individual arrays (one per image) as expected by SVM
        images = [ds.images[i] for i in range(len(ds))]
        y = ds.labels.astype(np.int64)
    else:
        # Fallback for other dataset types
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
