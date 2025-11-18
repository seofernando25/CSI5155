from pathlib import Path
from typing import Callable, Optional, Sequence, Sized
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from device import device
from utils.paths import require_file


def summarize_classification_results(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    loss: float | None = None,
) -> dict:
    if labels.size == 0:
        raise ValueError("No labels provided for evaluation summary.")

    accuracy = float(np.mean(predictions == labels))
    total = int(labels.size)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%) | Samples: {total}")
    if loss is not None:
        print(f"Loss: {loss:.4f}")

    per_class_accuracy: dict[str, float] = {}
    for idx, class_name in enumerate(class_names):
        class_mask = labels == idx
        class_count = int(np.sum(class_mask))
        if class_count > 0:
            class_accuracy = float(
                np.mean(predictions[class_mask] == labels[class_mask])
            )
            per_class_accuracy[class_name] = class_accuracy

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )

    cm = confusion_matrix(labels, predictions)

    macro_precision = precision_score(
        labels, predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    weighted_precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    weighted_recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "loss": loss,
        "per_class_accuracy": per_class_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm,
    }

    return metrics


def run_classification_evaluation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    class_names: Sequence[str],
    criterion: nn.Module | None = None,
    progress_desc: str = "Evaluating",
) -> dict:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=progress_desc):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total == 0:
        raise ValueError("Evaluation data loader returned zero samples.")

    all_predictions_arr = np.array(all_predictions)
    all_labels_arr = np.array(all_labels)

    test_accuracy = correct / total
    test_loss = running_loss / len(data_loader)

    summary = summarize_classification_results(
        labels=all_labels_arr,
        predictions=all_predictions_arr,
        class_names=class_names,
        loss=test_loss,
    )
    # ensure backwards compatibility for keys
    summary["accuracy"] = test_accuracy
    summary["loss"] = test_loss
    return summary


def evaluate_model_checkpoint(
    checkpoint_path: str | Path,
    model_builder: Callable[[dict, torch.device], torch.nn.Module],
    batch_size: int,
    class_names: Sequence[str],
    dataloader_factory: Callable[[str, int, bool], DataLoader],
    evaluation_fn: Callable[
        [torch.nn.Module, DataLoader, Sequence[str]], dict
    ] = run_classification_evaluation,
    on_checkpoint_loaded: Optional[Callable[[dict], None]] = None,
) -> dict:
    model_path_obj = require_file(checkpoint_path, hint="Train the model first")

    checkpoint = torch.load(str(model_path_obj), map_location=device)
    model = model_builder(checkpoint, device)

    if on_checkpoint_loaded is not None:
        on_checkpoint_loaded(checkpoint)

    test_loader = dataloader_factory("test", batch_size, False)
    # Get dataset size - assert dataset is Sized
    assert isinstance(test_loader.dataset, Sized), "Dataset must be Sized to get length"
    metrics = evaluation_fn(model, test_loader, class_names)

    return metrics


def run_checkpoint_evaluation_cli(
    model_path: str,
    batch_size: int,
    model_builder: Callable[[dict, torch.device], torch.nn.Module],
    class_names: Sequence[str],
    dataloader_factory: Callable[[str, int, bool], DataLoader],
    evaluation_fn: Callable[
        [torch.nn.Module, DataLoader, Sequence[str]], dict
    ] = run_classification_evaluation,
    on_checkpoint_loaded: Optional[Callable[[dict], None]] = None,
    missing_checkpoint_hint: str | None = None,
) -> dict:
    try:
        return evaluate_model_checkpoint(
            checkpoint_path=model_path,
            model_builder=model_builder,
            batch_size=batch_size,
            class_names=class_names,
            dataloader_factory=dataloader_factory,
            evaluation_fn=evaluation_fn,
            on_checkpoint_loaded=on_checkpoint_loaded,
        )
    except FileNotFoundError as exc:
        if missing_checkpoint_hint:
            model_path_obj = Path(model_path)
            raise FileNotFoundError(
                f"Model file not found at {model_path_obj}. {missing_checkpoint_hint}"
            ) from exc
        raise
