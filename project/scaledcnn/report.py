from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from device import device
from data import get_cifar10_class_names, get_cifar10_dataloader
from paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, TENSORBOARD_DIR
from scaledcnn.eval import build_model_from_checkpoint
from utils import (
    collect_scaledcnn_predictions,
    generate_classification_report_and_confusion_matrix,
)


@dataclass
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return 1.0 - self.values

    def summary(self) -> Dict[str, float]:
        if len(self.values) == 0:
            raise ValueError("Scalar series is empty.")
        errors = self.errors
        best_idx = int(np.argmin(errors))
        return {
            "final_error": float(errors[-1]),
            "min_error": float(errors.min()),
            "max_error": float(errors.max()),
            "best_step": int(self.steps[best_idx]),
        }


def _load_scalar_series(logdir: Path, tag: str) -> ScalarSeries:
    accumulator = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        raise ValueError(f"Tag '{tag}' not found in TensorBoard log {logdir}.")
    scalars = accumulator.Scalars(tag)
    steps = np.array([s.step for s in scalars], dtype=np.int64)
    values = np.array([s.value for s in scalars], dtype=np.float32)
    return ScalarSeries(steps=steps, values=values)


def run(
    k: int,
    split: str = "test",
    batch_size: int = 256,
):
    model_path_obj = MODELS_DIR / f"scaledcnn_k{k}.pth"
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found at {model_path_obj}")

    # Infer TensorBoard log directory from k value
    base_dir = TENSORBOARD_DIR / "training" / f"scaledcnn_k{k}"
    if not base_dir.exists():
        raise FileNotFoundError(f"No TensorBoard directory found at {base_dir}")
    runs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found inside {base_dir}")
    logdir_path = runs[-1]

    if not logdir_path.exists():
        raise FileNotFoundError(f"Log directory not found at {logdir_path}")

     # Set up output paths
    model_stem = model_path_obj.stem
    error_figure_path_obj = FIGURES_DIR / f"{model_stem}_error_curve.pdf"
    json_path_obj = METRICS_DIR / f"{model_stem}_{split}_training_report.json"

    train_series = _load_scalar_series(logdir_path, "train/accuracy")
    val_series = _load_scalar_series(logdir_path, "val/accuracy")
    smoothing_slider = 0.9

    def _exp_smooth(values: np.ndarray) -> np.ndarray:
        smoothed = np.empty_like(values, dtype=np.float32)
        last = values[0]
        for idx, point in enumerate(values):
            last = smoothing_slider * last + (1.0 - smoothing_slider) * point
            smoothed[idx] = last
        return smoothed

    train_smoothed = _exp_smooth(train_series.errors)
    val_smoothed = _exp_smooth(val_series.errors)
    plt.figure(figsize=(6, 4))
    plt.plot(
        train_series.steps,
        train_series.errors,
        label=None,
        alpha=0.3,
        color="#1f77b4",
    )
    plt.plot(
        val_series.steps,
        val_series.errors,
        label=None,
        alpha=0.3,
        color="#ff7f0e",
    )
    plt.plot(
        train_series.steps,
        train_smoothed,
        label="Train Error",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        val_series.steps,
        val_smoothed,
        label="Validation Error",
        color="#ff7f0e",
        linewidth=2,
    )
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(error_figure_path_obj), format="pdf")
    plt.close()

    # Generate classification report and predictions
    class_names = get_cifar10_class_names()
    checkpoint = torch.load(str(model_path_obj), map_location=device)
    model, config = build_model_from_checkpoint(checkpoint, device)

    data_loader = get_cifar10_dataloader(
        split=split,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions_arr, labels_arr = collect_scaledcnn_predictions(model, data_loader)
    
    # Generate confusion matrix, classification report, and save metrics
    k = config.get("k", 1)
    model_token = f"scaledcnn_k{k}"
    
    result = generate_classification_report_and_confusion_matrix(
        labels=labels_arr,
        predictions=predictions_arr,
        class_names=class_names,
        model_token=model_token,
        split=split,
    )
    classification_report_dict = result["classification_report"]

    payload = {
        "model_path": str(model_path_obj),
        "logdir": str(logdir_path),
        "split": split,
        "config": config,
        "training_error": {
            "train": train_series.summary(),
            "validation": val_series.summary(),
        },
        "classification_report": classification_report_dict,
        "error_figure": str(error_figure_path_obj),
        "confusion_figure": result["confusion_figure"],
    }

    with json_path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "report",
        help="Generate training error stats, classification report, and confusion matrix for a ScaledCNN checkpoint",
    )
    parser.add_argument("--k", type=int, required=True, help="Scaling factor k")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split for evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.set_defaults(entry=lambda args: run(k=args.k, split=args.split, batch_size=args.batch_size))
    return parser


