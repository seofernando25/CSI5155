import shutil
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk

RANDOM_SEED = 6920


def process_dataset(
    src_dir: Path, dst_dir: Path
) -> None:
    ds_dict = load_from_disk(str(src_dir))
    assert isinstance(ds_dict, DatasetDict), "ds_dict must be a DatasetDict"
    dst_dir.mkdir(parents=True, exist_ok=True)

    def normalize_images(examples):
        normalized = []
        for img in examples["img"]:
            normalized.append(np.asarray(img, dtype=np.float32) / 255.0)
        return {"img": normalized}

    normalized_ds_dict = DatasetDict({
        split: ds.map(normalize_images, batched=True, desc=f"Normalizing {split}")
        for split, ds in ds_dict.items()
    })

    train_ds = normalized_ds_dict["train"]
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(len(train_ds))
    
    split_idx = int(len(train_ds) * 0.8)
    train_split = train_ds.select(indices[:split_idx])
    val_split = train_ds.select(indices[split_idx:])
    
    print(f"Split training set: {len(train_split)} train, {len(val_split)} validation")

    noise_rate = 0.2
    num_corrupt = int(len(train_split) * noise_rate)
    labels = np.array(train_split["label"])
    corrupt_indices = rng.choice(len(train_split), size=num_corrupt, replace=False)
    corrupted_labels = labels.copy()

    corrupted_labels[corrupt_indices] = (labels[corrupt_indices] + rng.randint(1, 10, size=num_corrupt)) % 10
    assert np.all(corrupted_labels[corrupt_indices] != labels[corrupt_indices]), "Corruption failed: some labels unchanged!"

    original_label_feature = train_split.features["label"]
    train_split = train_split.remove_columns(["label"]).add_column("label", corrupted_labels.tolist())
    train_split = train_split.cast_column("label", original_label_feature)
    
    print(f"Poisoned {num_corrupt} samples ({noise_rate*100:.1f}%) in training set")

    DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": normalized_ds_dict["test"],
    }).save_to_disk(str(dst_dir))


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
    process_dataset(src, dst)
    print("Done.")
