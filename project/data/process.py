import shutil
from pathlib import Path
from typing import Callable, List

import numpy as np
from datasets import DatasetDict, load_from_disk
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import img_as_float


def build_transform_pipeline():
    def transform_one(image) -> np.ndarray:
        arr = np.asarray(image)
        arr = rgb2gray(arr) if arr.ndim == 3 else arr
        arr = img_as_float(arr)
        arr = resize(arr, (32, 32), anti_aliasing=True, preserve_range=True)
        return arr

    def transform_batch(images: List) -> List:
        return [transform_one(img) for img in images]

    return transform_batch


def process_dataset(
    src_dir: Path, dst_dir: Path, transform_batch: Callable[[List], List]
) -> None:
    ds_dict: DatasetDict = load_from_disk(str(src_dir))
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split, ds in ds_dict.items():
        ds_dict[split] = ds.map(
            lambda batch: {"img": transform_batch(batch["img"])},
            batched=True,
            num_proc=4,
        )

    ds_dict.save_to_disk(str(dst_dir))


def run(force: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root / ".cache" / "base_datasets"
    out_root = repo_root / ".cache" / "processed_datasets"
    out_root.mkdir(parents=True, exist_ok=True)

    src = base_root / "cifar10"
    dst = out_root / "cifar10"
    if dst.exists() and force:
        shutil.rmtree(dst, ignore_errors=True)

    print(f"Processing CIFAR-10: {src} -> {dst}")
    transform_batch = build_transform_pipeline()
    process_dataset(src, dst, transform_batch)
    print("Done.")
