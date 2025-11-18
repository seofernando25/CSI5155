import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from mlxtend.evaluate import mcnemar

from device import device
from data.datasets import get_cifar10_dataloader, get_cifar10_split
from paths import METRICS_DIR, REPO_ROOT
from scaledcnn.eval import build_model_from_checkpoint
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM
from utils.metrics import collect_scaledcnn_predictions
from utils.paths import require_file


DEFAULT_RESULTS_PATH = METRICS_DIR / "mcnemar_results_mlxtend.json"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    model_path: Path


def _format_pair_token(token: str) -> Tuple[str, str]:
    left, sep, right = token.partition(":")
    if not sep:
        raise ValueError(
            f"Invalid pair specification '{token}'. Expected format modelA:modelB"
        )
    return left.strip(), right.strip()


def main(argv: List[str] | None = None) -> Dict:
    parser = argparse.ArgumentParser(
        description="Run McNemar's tests between model prediction pairs using mlxtend."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Model pairs in the form modelA:modelB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for ScaledCNN inference (default: 256)",
    )
    args = parser.parse_args(argv)

    specs: Dict[str, ModelSpec] = {}
    predictions_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pair_token in args.pairs:
        model_a_id, model_b_id = _format_pair_token(pair_token)
        for model_id in (model_a_id, model_b_id):
            if model_id not in specs:
                if model_id == "svm":
                    model_path = (REPO_ROOT / Path(SVM_CLASSIFIER_PATH)).resolve()
                elif model_id.startswith("scaledcnn_k"):
                    model_path = (
                        REPO_ROOT / ".cache" / "models" / f"{model_id}.pth"
                    ).resolve()
                else:
                    raise ValueError(
                        f"Unknown model id '{model_id}'. Provide --model-path {model_id}=/path/to/model."
                    )

                model_type = "svm" if model_id == "svm" else "scaledcnn"
                model_path = require_file(model_path)

                specs[model_id] = ModelSpec(
                    model_id=model_id, model_type=model_type, model_path=model_path
                )

                if model_type == "svm":
                    # Compute SVM predictions
                    split = "test"
                    print(f"[svm] Loading model from {model_path}")
                    model = ClassifierSVM.load(str(model_path))
                    images, labels = get_cifar10_split(split)
                    print(
                        f"[svm] Computing predictions on {split} split ({len(images)} samples)"
                    )
                    predictions = model.predict(images).astype(np.int64)
                    labels_arr = labels.astype(np.int64)
                    predictions_cache[model_id] = (predictions, labels_arr)
                else:
                    # Compute ScaledCNN predictions
                    split = "test"
                    require_file(model_path)

                    checkpoint = torch.load(str(model_path), map_location=device)
                    model, _ = build_model_from_checkpoint(checkpoint, device)

                    data_loader = get_cifar10_dataloader(
                        split=split,
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                    total_samples = len(data_loader.dataset)
                    print(
                        f"[scaledcnn] Computing predictions on {split} split ({total_samples} samples)"
                    )

                    predictions, labels = collect_scaledcnn_predictions(
                        model, data_loader
                    )
                    predictions_cache[model_id] = (predictions, labels)

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

        correct_a = preds_a == labels_a
        correct_b = preds_b == labels_a

        n01 = int(np.sum(~correct_a & correct_b))
        n10 = int(np.sum(correct_a & ~correct_b))
        n11 = int(np.sum(correct_a & correct_b))
        n00 = int(np.sum(~correct_a & ~correct_b))

        contingency = np.array([[n11, n10], [n01, n00]], dtype=np.int64)
        discordant = n01 + n10
        assert discordant >= 25, "McNemar's test requires at least 25 discordant pairs."
        chi2, p_value = mcnemar(ary=contingency)
        chi_square = float(chi2) if chi2 is not None else None

        stats = {
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "n00": n00,
            "discordant": discordant,
            "chi_square": chi_square,
            "p_value": float(p_value),
        }
        result = {
            "model_a": model_a_id,
            "model_b": model_b_id,
            **stats,
        }
        pair_results.append(result)
        chi_square_val = stats["chi_square"]
        chi_square_repr = (
            f"{chi_square_val:.4f}" if chi_square_val is not None else "None"
        )
        print(
            f"[mcnemar] {model_a_id} vs {model_b_id}: "
            f"chi-square={chi_square_repr}, p-value={stats['p_value']:.3g}"
        )

    output_path = DEFAULT_RESULTS_PATH
    payload = {
        "split": "test",
        "pairs": pair_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[mcnemar] Results saved to {output_path}")
    return payload


if __name__ == "__main__":
    main()
