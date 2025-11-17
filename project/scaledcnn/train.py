from pathlib import Path

from datetime import datetime
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, TensorDataset

from scaledcnn.training import run_training_loop
from scaledcnn.model import ScaledCNN


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(
    model_path: str | None = None,
    k: int = 1,
    device: str | None = None,
    resume: bool = True,
):
    epochs = 500
    batch_size = 128
    set_seed(k)

    if model_path is None:
        model_path = f".cache/models/scaledcnn_k{k}.pth"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/training/scaledcnn_k{k}/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir.resolve()}")

    # Define custom scalars layout to group train/val metrics together
    layout = {
        "Loss": {
            "Train vs Val Loss": ["Multiline", ["train/loss", "val/loss"]],
        },
        "Accuracy": {
            "Train vs Val Accuracy": ["Multiline", ["train/accuracy", "val/accuracy"]],
        },
    }
    writer.add_custom_scalars(layout)

    torch_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {torch_device}")

    # Enable cuDNN benchmark for faster convolutions
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("Pre-loading dataset into GPU memory...")
    print("Loading training data...")
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "processed_datasets" / "cifar10"

    train_images_np = np.load(dataset_path / "train_images.npy", mmap_mode="r")
    train_images = (
        torch.from_numpy(train_images_np.copy()).permute(0, 3, 1, 2).to(torch_device)
    )
    train_labels_np = np.load(dataset_path / "train_labels.npy", mmap_mode="r")
    train_labels = torch.from_numpy(train_labels_np.copy()).to(torch_device)
    print(
        f"Loaded {len(train_images)} training images to {torch_device} ({train_images.element_size() * train_images.nelement() / 1024**2:.1f} MB)"
    )

    print("Loading validation data...")
    val_images_np = np.load(dataset_path / "validation_images.npy", mmap_mode="r")
    val_images = (
        torch.from_numpy(val_images_np.copy()).permute(0, 3, 1, 2).to(torch_device)
    )
    val_labels_np = np.load(dataset_path / "validation_labels.npy", mmap_mode="r")
    val_labels = torch.from_numpy(val_labels_np.copy()).to(torch_device)
    print(
        f"Loaded {len(val_images)} validation images to {torch_device} ({val_images.element_size() * val_images.nelement() / 1024**2:.1f} MB)"
    )

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"\nInitializing ScaledCNN(k={k}) model...")
    model = ScaledCNN(
        k=k,
        num_classes=10,
    ).to(torch_device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
    )

    start_time = time.time()
    print(f"\nTraining for {epochs} epochs...")
    model_config = {
        "k": k,
        "num_classes": 10,
    }

    def _checkpoint_extra(
        _epoch: int,
        _train_loss: float,
        _train_acc: float,
        _val_loss: float,
        _val_acc: float,
    ) -> dict[str, object]:
        return {"config": model_config}

    training_result = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch_device,
        epochs=epochs,
        checkpoint_path=model_path,
        writer=writer,
        checkpoint_extra=_checkpoint_extra,
        resume=resume,
    )

    elapsed = time.time() - start_time
    print("\nTraining complete!")
    best_val_acc = training_result.best_val_accuracy
    final_train_loss = training_result.final_train_loss
    final_train_acc = training_result.final_train_accuracy
    final_val_loss = training_result.final_val_loss
    final_val_acc = training_result.final_val_accuracy
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Total training time: {elapsed / 60:.2f} minutes")

    hparams = {
        "k": k,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "optimizer": "Adam",
    }
    metrics = {
        "hparam/best_val_accuracy": best_val_acc,
        "hparam/final_val_accuracy": final_val_acc,
        "hparam/final_train_accuracy": final_train_acc,
        "hparam/final_val_loss": final_val_loss,
        "hparam/final_train_loss": final_train_loss,
        "hparam/time_minutes": elapsed / 60,
    }
    writer.add_hparams(hparams, metrics)
    writer.close()

    return best_val_acc, log_dir


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", help="ScaledCNN train")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model path (default: .cache/models/scaledcnn_k{k}.pth)",
    )
    parser.add_argument("--k", type=int, default=1, help="Scaling factor k")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch even if checkpoint exists",
    )

    def _entry(args):
        return run(
            model_path=args.model_path,
            k=args.k,
            device=args.device,
            resume=not args.no_resume,
        )

    parser.set_defaults(entry=_entry)
    return parser
