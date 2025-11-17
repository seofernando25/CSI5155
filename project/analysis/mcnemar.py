from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from data import get_cifar10_dataloader, get_cifar10_split
from scaledcnn.confusion_matrix import build_model_from_checkpoint
from scaledcnn.evaluation import resolve_torch_device
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / ".cache" / "predictions"
METRICS_DIR = REPO_ROOT / ".cache" / "metrics"
DEFAULT_RESULTS_PATH = METRICS_DIR / "mcnemar_results.json"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    model_path: Path


def _ensure_directories() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _prediction_cache_path(model_id: str, split: str) -> Path:
    safe_id = model_id.replace("/", "_")
    return PREDICTIONS_DIR / f"{safe_id}_{split}.npz"


def _load_cached_predictions(cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return data["predictions"].astype(np.int64), data["labels"].astype(np.int64)


def _save_predictions(
    cache_path: Path, predictions: np.ndarray, labels: np.ndarray, metadata: Dict
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        predictions=predictions.astype(np.int64),
        labels=labels.astype(np.int64),
        metadata=json.dumps(metadata),
    )


def _compute_svm_predictions(
    spec: ModelSpec,
    split: str,
    force: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_path = _prediction_cache_path(spec.model_id, split)
    if cache_path.exists() and not force:
        print(f"[cache] Loading SVM predictions from {cache_path}")
        return _load_cached_predictions(cache_path)

    print(f"[svm] Loading model from {spec.model_path}")
    model = ClassifierSVM.load(str(spec.model_path))
    images, labels = get_cifar10_split(split)
    print(f"[svm] Computing predictions on {split} split ({len(images)} samples)")
    predictions = model.predict(images).astype(np.int64)
    labels_arr = labels.astype(np.int64)
    _save_predictions(
        cache_path,
        predictions,
        labels_arr,
        {"model_id": spec.model_id, "model_path": str(spec.model_path), "split": split},
    )
    print(f"[svm] Saved predictions to {cache_path}")
    return predictions, labels_arr


def _compute_scaledcnn_predictions(
    spec: ModelSpec,
    split: str,
    batch_size: int,
    device: str | None,
    force: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_path = _prediction_cache_path(spec.model_id, split)
    if cache_path.exists() and not force:
        print(f"[cache] Loading ScaledCNN predictions from {cache_path}")
        return _load_cached_predictions(cache_path)

    torch_device = resolve_torch_device(device)
    print(f"[scaledcnn] Using device: {torch_device}")
    if not spec.model_path.exists():
        raise FileNotFoundError(f"ScaledCNN checkpoint not found at {spec.model_path}")

    checkpoint = torch.load(str(spec.model_path), map_location=torch_device)
    model, _ = build_model_from_checkpoint(checkpoint, torch_device)
    model.eval()

    data_loader, _ = get_cifar10_dataloader(
        split=split,
        batch_size=batch_size,
        device=torch_device,
        shuffle=False,
    )
    total_samples = len(data_loader.dataset)
    print(f"[scaledcnn] Computing predictions on {split} split ({total_samples} samples)")

    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(torch_device)
            batch_labels = batch_labels.to(torch_device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(all_predictions).astype(np.int64)
    labels = np.concatenate(all_labels).astype(np.int64)
    _save_predictions(
        cache_path,
        predictions,
        labels,
        {"model_id": spec.model_id, "model_path": str(spec.model_path), "split": split},
    )
    print(f"[scaledcnn] Saved predictions to {cache_path}")
    return predictions, labels


def _resolve_model_spec(
    model_id: str, custom_paths: Dict[str, Path]
) -> ModelSpec:
    if model_id in custom_paths:
        model_path = custom_paths[model_id]
    elif model_id == "svm":
        model_path = (REPO_ROOT / Path(SVM_CLASSIFIER_PATH)).resolve()
    elif model_id.startswith("scaledcnn_k"):
        model_path = (REPO_ROOT / ".cache" / "models" / f"{model_id}.pth").resolve()
    else:
        raise ValueError(
            f"Unknown model id '{model_id}'. Provide --model-path {model_id}=/path/to/model."
        )

    model_type = "svm" if model_id == "svm" else "scaledcnn"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for '{model_id}' at {model_path}")
    return ModelSpec(model_id=model_id, model_type=model_type, model_path=model_path)


def _parse_model_overrides(entries: Iterable[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid --model-path override '{entry}'. Expected format id=/abs/path"
            )
        overrides[key.strip()] = Path(value.strip()).expanduser().resolve()
    return overrides


def _get_predictions_for_spec(
    spec: ModelSpec,
    split: str,
    batch_size: int,
    device: str | None,
    force: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if spec.model_type == "svm":
        return _compute_svm_predictions(spec, split, force)
    if spec.model_type == "scaledcnn":
        return _compute_scaledcnn_predictions(spec, split, batch_size, device, force)
    raise ValueError(f"Unsupported model type '{spec.model_type}'")


def run_mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float | int]:
    if not (len(predictions_a) == len(predictions_b) == len(labels)):
        raise ValueError("Predictions and labels must have the same length.")

    correct_a = predictions_a == labels
    correct_b = predictions_b == labels

    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))

    discordant = n01 + n10
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(n01 - n10) - 1) ** 2) / discordant
        p_value = float(math.erfc(math.sqrt(statistic / 2.0)))

    return {
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n00": n00,
        "discordant": discordant,
        "chi_square": float(statistic),
        "p_value": p_value,
    }


def _format_pair_token(token: str) -> Tuple[str, str]:
    left, sep, right = token.partition(":")
    if not sep:
        raise ValueError(
            f"Invalid pair specification '{token}'. Expected format modelA:modelB"
        )
    return left.strip(), right.strip()


def main(argv: List[str] | None = None) -> Dict:
    parser = argparse.ArgumentParser(
        description="Run McNemar's tests between model prediction pairs."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Model pairs in the form modelA:modelB (e.g., svm:scaledcnn_k64)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for ScaledCNN inference (default: 256)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for ScaledCNN inference (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions even if cached files exist.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Override model path mapping, e.g., scaledcnn_k64=/abs/path/to/checkpoint.pth",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        help="Output JSON path for McNemar results.",
    )
    args = parser.parse_args(argv)

    _ensure_directories()
    custom_paths = _parse_model_overrides(args.model_path)

    specs: Dict[str, ModelSpec] = {}
    predictions_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pair_token in args.pairs:
        model_a_id, model_b_id = _format_pair_token(pair_token)
        for model_id in (model_a_id, model_b_id):
            if model_id not in specs:
                specs[model_id] = _resolve_model_spec(model_id, custom_paths)
                predictions_cache[model_id] = _get_predictions_for_spec(
                    specs[model_id],
                    split=args.split,
                    batch_size=args.batch_size,
                    device=args.device,
                    force=args.force,
                )

    pair_results = []
    for pair_token in args.pairs:
        model_a_id, model_b_id = _format_pair_token(pair_token)
        preds_a, labels_a = predictions_cache[model_a_id]
        preds_b, labels_b = predictions_cache[model_b_id]
        if not np.array_equal(labels_a, labels_b):
            raise ValueError(
                f"Label vectors for {model_a_id} and {model_b_id} do not match. "
                "Ensure both predictions were computed on the same dataset split."
            )
        stats = run_mcnemar_test(preds_a, preds_b, labels_a)
        result = {
            "model_a": model_a_id,
            "model_b": model_b_id,
            **stats,
        }
        pair_results.append(result)
        print(
            f"[mcnemar] {model_a_id} vs {model_b_id}: "
            f"chi-square={stats['chi_square']:.4f}, p-value={stats['p_value']:.3}"
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "pairs": pair_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[mcnemar] Results saved to {output_path}")
    return payload


if __name__ == "__main__":
    main()


