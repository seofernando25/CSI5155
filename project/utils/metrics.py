from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from device import device
from paths import FIGURES_DIR, METRICS_DIR
from visualization.confusion import plot_confusion_matrix


def save_confusion_matrix_json(
    labels: np.ndarray,
    predictions: np.ndarray,
    model_token: str,
    split: str,
    class_names: Sequence[str],
) -> Path:
    raw_cm = confusion_matrix(labels, predictions)
    row_sums = raw_cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    normalized_cm = raw_cm.astype(np.float64) / row_sums

    confusion_matrix_path = METRICS_DIR / f"{model_token}_{split}_confusion_matrix.json"
    confusion_matrix_payload = {
        "model": model_token,
        "split": split,
        "class_names": class_names,
        "matrix": raw_cm.tolist(),
        "normalized_matrix": normalized_cm.tolist(),
    }
    with confusion_matrix_path.open("w", encoding="utf-8") as f:
        json.dump(confusion_matrix_payload, f, indent=2)
    print(f"Confusion matrix counts saved to: {confusion_matrix_path}")
    return confusion_matrix_path


def collect_scaledcnn_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(images)
            batch_preds = torch.argmax(outputs, dim=1)
            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    predictions_arr = np.array(predictions, dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.int64)
    return predictions_arr, labels_arr


def generate_classification_report_and_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    model_token: str,
    split: str,
) -> dict:
    confusion_figure_path = (
        FIGURES_DIR / f"{model_token}__{split}__confusion_matrix.pdf"
    )
    metrics_json_path = METRICS_DIR / f"{model_token}_{split}_metrics.json"

    plot_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        output_path=str(confusion_figure_path),
    )
    print(f"Confusion matrix figure saved to: {confusion_figure_path}")

    save_confusion_matrix_json(
        labels=labels,
        predictions=predictions,
        model_token=model_token,
        split=split,
        class_names=class_names,
    )
    classification_report_dict = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(classification_report_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

    return {
        "classification_report": classification_report_dict,
        "confusion_figure": str(confusion_figure_path),
        "metrics_json": str(metrics_json_path),
    }
