from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class TrainingLoopResult:
    best_val_accuracy: float
    final_train_loss: float
    final_train_accuracy: float
    final_val_loss: float
    final_val_accuracy: float
    epochs: int
    checkpoint_path: Path


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
    progress_desc: str = "Training",
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc=progress_desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total if total else 0.0

    if writer is not None and epoch is not None:
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
    progress_desc: str = "Validating",
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc=progress_desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total if total else 0.0

    if writer is not None and epoch is not None:
        writer.add_scalar("val/loss", epoch_loss, epoch)
        writer.add_scalar("val/accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    writer: Optional[SummaryWriter] = None,
    log_lr: bool = True,
    log_param_histograms: bool = True,
    checkpoint_extra: Optional[
        Callable[[int, float, float, float, float], dict[str, Any] | None]
    ] = None,
    train_progress_desc: str = "Training",
    val_progress_desc: str = "Validating",
) -> TrainingLoopResult:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    final_train_loss = 0.0
    final_train_acc = 0.0
    final_val_loss = 0.0
    final_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch,
            progress_desc=train_progress_desc,
        )
        final_train_loss = train_loss
        final_train_acc = train_acc
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)"
        )

        val_loss, val_acc = validate_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch,
            progress_desc=val_progress_desc,
        )
        final_val_loss = val_loss
        final_val_acc = val_acc
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc * 100:.2f}%)"
        )

        if writer is not None and log_lr:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if writer is not None and log_param_histograms:
            for name, param in model.named_parameters():
                writer.add_histogram(f"params/{name}", param.detach().cpu(), epoch)
                if param.grad is not None:
                    writer.add_histogram(
                        f"grads/{name}", param.grad.detach().cpu(), epoch
                    )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_data: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
            }
            if checkpoint_extra is not None:
                extra = checkpoint_extra(
                    epoch, train_loss, train_acc, val_loss, val_acc
                )
                if extra:
                    checkpoint_data.update(extra)

            torch.save(checkpoint_data, str(checkpoint_path))
            print(f"Saved best model (val_acc={val_acc:.4f}) to {checkpoint_path}")

    return TrainingLoopResult(
        best_val_accuracy=best_val_acc,
        final_train_loss=final_train_loss,
        final_train_accuracy=final_train_acc,
        final_val_loss=final_val_loss,
        final_val_accuracy=final_val_acc,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
    )
