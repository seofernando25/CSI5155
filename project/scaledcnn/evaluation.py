from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Sized, Union

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

def resolve_torch_device(device: str | None) -> torch.device:
    return (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def summarize_classification_results(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    loss: float | None = None,
    evaluation_time: float | None = None,
    prediction_time: float | None = None,
) -> dict:
    if labels.size == 0:
        raise ValueError("No labels provided for evaluation summary.")

    accuracy = float(np.mean(predictions == labels))
    total = int(labels.size)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if loss is not None:
        print(f"Test loss: {loss:.4f}")
    print(f"Test samples: {total}")
    if evaluation_time is not None:
        print(f"Evaluation time: {evaluation_time:.2f} seconds")
        print(f"Time per sample: {(evaluation_time / total) * 1000:.2f} ms")
    if prediction_time is not None:
        print(f"Prediction time: {prediction_time:.2f} seconds")

    per_class_accuracy: dict[str, float] = {}
    print("\nPer-class accuracy:")
    for idx, class_name in enumerate(class_names):
        class_mask = labels == idx
        class_count = int(np.sum(class_mask))
        if class_count > 0:
            class_accuracy = float(
                np.mean(predictions[class_mask] == labels[class_mask])
            )
            per_class_accuracy[class_name] = class_accuracy
            print(
                f"  {class_name:12s}: {class_accuracy:.4f} ({class_accuracy * 100:.2f}%) "
                f"[{class_count} samples]"
            )

    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )
    print(report)

    cm = confusion_matrix(labels, predictions)
    print("=" * 60)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 60)
    print(f"Total correct predictions: {np.trace(cm)} / {total}")
    print(f"Total incorrect predictions: {total - np.trace(cm)}")

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

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Macro Avg':<15} {'Weighted Avg':<15}")
    print("-" * 60)
    print(f"{'Precision':<20} {macro_precision:<15.4f} {weighted_precision:<15.4f}")
    print(f"{'Recall':<20} {macro_recall:<15.4f} {weighted_recall:<15.4f}")
    print(f"{'F1-Score':<20} {macro_f1:<15.4f} {weighted_f1:<15.4f}")
    print("=" * 60)

    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "loss": loss,
        "evaluation_time": evaluation_time,
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

    if prediction_time is not None:
        metrics["prediction_time"] = prediction_time

    return metrics


def run_classification_evaluation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
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

    start_time = time.time()
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

    eval_time = time.time() - start_time
    all_predictions_arr = np.array(all_predictions)
    all_labels_arr = np.array(all_labels)

    test_accuracy = correct / total
    test_loss = running_loss / len(data_loader)

    summary = summarize_classification_results(
        labels=all_labels_arr,
        predictions=all_predictions_arr,
        class_names=class_names,
        loss=test_loss,
        evaluation_time=eval_time,
    )
    # ensure backwards compatibility for keys
    summary["accuracy"] = test_accuracy
    summary["loss"] = test_loss
    summary["evaluation_time"] = eval_time
    return summary


def evaluate_model_checkpoint(
    checkpoint_path: str | Path,
    model_builder: Callable[[dict, torch.device], torch.nn.Module],
    batch_size: int,
    device: str | None,
    class_names: Sequence[str],
    dataloader_factory: Callable[
        [str, int, Union[torch.device, str, None], bool], tuple[DataLoader, torch.device]
    ],
    evaluation_fn: Callable[
        [torch.nn.Module, DataLoader, torch.device, Sequence[str]], dict
    ] = run_classification_evaluation,
    on_checkpoint_loaded: Optional[Callable[[dict], None]] = None,
) -> dict:
    torch_device = resolve_torch_device(device)
    print(f"Using device: {torch_device}")

    model_path_obj = Path(checkpoint_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path_obj}. Please train the model first."
        )

    print(f"Loading model from: {model_path_obj}")
    checkpoint = torch.load(str(model_path_obj), map_location=torch_device)
    model = model_builder(checkpoint, torch_device)

    print("Model loaded successfully!")
    if "epoch" in checkpoint:
        print(f"Model was trained for {checkpoint['epoch']} epochs")
    if "val_acc" in checkpoint:
        print(
            f"Model validation accuracy: {checkpoint['val_acc']:.4f} "
            f"({checkpoint['val_acc'] * 100:.2f}%)"
        )

    if on_checkpoint_loaded is not None:
        on_checkpoint_loaded(checkpoint)

    print("\nLoading CIFAR-10 test dataset...")
    test_loader, torch_device = dataloader_factory("test", batch_size, torch_device, False)
    # Get dataset size - assert dataset is Sized
    assert isinstance(test_loader.dataset, Sized), "Dataset must be Sized to get length"
    dataset_size = len(test_loader.dataset)
    print(f"Test samples: {dataset_size}")

    print("\nEvaluating model on test set...")
    metrics = evaluation_fn(model, test_loader, torch_device, class_names)

    return metrics


def run_checkpoint_evaluation_cli(
    model_path: str,
    batch_size: int,
    device: str | None,
    model_builder: Callable[[dict, torch.device], torch.nn.Module],
    class_names: Sequence[str],
    dataloader_factory: Callable[
        [str, int, Union[torch.device, str, None], bool], tuple[DataLoader, torch.device]
    ],
    evaluation_fn: Callable[
        [torch.nn.Module, DataLoader, torch.device, Sequence[str]], dict
    ] = run_classification_evaluation,
    on_checkpoint_loaded: Optional[Callable[[dict], None]] = None,
    missing_checkpoint_hint: str | None = None,
) -> dict:
    try:
        return evaluate_model_checkpoint(
            checkpoint_path=model_path,
            model_builder=model_builder,
            batch_size=batch_size,
            device=device,
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

