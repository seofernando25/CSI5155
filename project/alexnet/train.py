from pathlib import Path

from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from alexnet.model import CIFARAlexNet
from common_net.training import build_dataloader, run_training_loop
from data import CIFAR10Dataset, load_cifar10_data, prepare_split


def run(
    model_path: str = ".cache/models/alexnet_cifar.pth",
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 0.075,
    device: str | None = None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/training/alexnet/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"TensorBoard logging to: {log_dir.resolve()}")

    torch_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {torch_device}")

    print("Loading CIFAR-10 dataset...")
    ds_dict = load_cifar10_data()

    print("Preparing training data...")
    X_all, y_all = prepare_split(ds_dict, "train")
    print(f"Total training samples: {len(X_all)}")

    n_train = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:], y_all[n_train:]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_dataset = CIFAR10Dataset(X_train, y_train, transform=None)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        device=torch_device,
        shuffle=True,
    )

    val_dataset = CIFAR10Dataset(X_val, y_val, transform=None)
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        device=torch_device,
        shuffle=False,
    )

    print("\nInitializing AlexNet model...")
    model = CIFARAlexNet(num_classes=10).to(torch_device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    print(f"\nTraining for {epochs} epochs...")
    training_result = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch_device,
        epochs=epochs,
        checkpoint_path=model_path,
        writer=writer,
        grad_clip_max_norm=1.0,
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

    if writer:
        hparams = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 2e-3,
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
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
    parser = subparsers.add_parser("train", help="AlexNet train")
    parser.add_argument("--model-path", default=".cache/models/alexnet_cifar.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.075)
    parser.add_argument("--device", type=str, default=None)

    def _entry(args):
        return run(
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

    parser.set_defaults(entry=_entry)
    return parser
