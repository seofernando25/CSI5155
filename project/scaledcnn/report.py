from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from tensorboard.backend.event_processing import event_accumulator

from data import get_cifar10_class_names, get_cifar10_dataloader
from scaledcnn.eval import build_model_from_checkpoint
from scaledcnn.evaluation import resolve_torch_device


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


def _default_paths(model_path: Path, split: str) -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    figures_dir = repo_root / ".cache" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = repo_root / ".cache" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_stem = model_path.stem
    figure_path = figures_dir / f"{model_stem}_error_curve.pdf"
    json_path = metrics_dir / f"{model_stem}_{split}_training_report.json"
    return figure_path, json_path


def _infer_logdir(model_path: Path, explicit_logdir: Optional[Path]) -> Path:
    if explicit_logdir is not None:
        return explicit_logdir

    repo_root = Path(__file__).resolve().parents[1]
    k_token = None
    stem = model_path.stem
    if "k" in stem:
        idx = stem.rfind("k")
        suffix = stem[idx + 1 :]
        digits = []
        for ch in suffix:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            k_token = "".join(digits)
    if k_token is None:
        raise ValueError(
            f"Could not infer k value from model path '{model_path}'. "
            "Please provide --logdir explicitly."
        )

    base_dir = repo_root / ".cache" / "tensorboard" / "training" / f"scaledcnn_k{k_token}"
    if not base_dir.exists():
        raise FileNotFoundError(f"No TensorBoard directory found at {base_dir}")
    runs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found inside {base_dir}")
    return runs[-1]


def _generate_classification_report(
    model_path: Path,
    split: str,
    batch_size: int,
    device: str | None,
) -> Tuple[Dict, Dict]:
    class_names = get_cifar10_class_names()
    torch_device = resolve_torch_device(device)
    checkpoint = torch.load(str(model_path), map_location=torch_device)
    model, config = build_model_from_checkpoint(checkpoint, torch_device)

    data_loader, _ = get_cifar10_dataloader(
        split=split,
        batch_size=batch_size,
        device=torch_device,
        shuffle=False,
    )

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(torch_device)
            batch_labels = batch_labels.to(torch_device)
            outputs = model(images)
            batch_preds = torch.argmax(outputs, dim=1)
            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    predictions_arr = np.array(predictions, dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.int64)
    report = classification_report(
        labels_arr,
        predictions_arr,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    return report, config


def _plot_error_curves(
    train_series: ScalarSeries,
    val_series: ScalarSeries,
    output_path: Path,
    model_label: str,
) -> None:
    smoothing_slider = 0.9  # TensorBoard-style decay on previous value

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), format="pdf")
    plt.close()


def run(
    model_path: str,
    logdir: str | None = None,
    split: str = "test",
    batch_size: int = 256,
    device: str | None = None,
    output_json: str | None = None,
    figure_path: str | None = None,
):
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found at {model_path_obj}")

    logdir_path = _infer_logdir(model_path_obj, Path(logdir) if logdir else None)
    if not logdir_path.exists():
        raise FileNotFoundError(f"Log directory not found at {logdir_path}")

    default_figure, default_json = _default_paths(model_path_obj, split)
    figure_path_obj = Path(figure_path) if figure_path else default_figure
    json_path_obj = Path(output_json) if output_json else default_json

    train_series = _load_scalar_series(logdir_path, "train/accuracy")
    val_series = _load_scalar_series(logdir_path, "val/accuracy")
    _plot_error_curves(train_series, val_series, figure_path_obj, model_path_obj.stem)

    classification_report_dict, config = _generate_classification_report(
        model_path=model_path_obj,
        split=split,
        batch_size=batch_size,
        device=device,
    )

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
        "figure": str(figure_path_obj),
    }

    json_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with json_path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved error plot to: {figure_path_obj}")
    print(f"Saved metrics report to: {json_path_obj}")
    return payload


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "report",
        help="Generate training error stats and classification report for a ScaledCNN checkpoint",
    )
    parser.add_argument("--model-path", default=".cache/models/scaledcnn.pth")
    parser.add_argument(
        "--logdir",
        default=None,
        help="TensorBoard log directory for the run (auto-inferred from model path if omitted).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split for evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for metrics JSON.",
    )
    parser.add_argument(
        "--figure-path",
        type=str,
        default=None,
        help="Optional output path for the error curve PDF.",
    )

    def _entry(args):
        return run(
            model_path=args.model_path,
            logdir=args.logdir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            output_json=args.output_json,
            figure_path=args.figure_path,
        )

    parser.set_defaults(entry=_entry)
    return parser


