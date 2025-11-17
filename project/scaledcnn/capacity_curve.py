from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scaledcnn.info import describe_model


def _load_metrics_for_k(k: int) -> Tuple[float, float]:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = repo_root / ".cache" / "metrics" / f"scaledcnn_k{k}_test_training_report.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found at {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    acc = float(data["classification_report"]["accuracy"])
    macro_f1 = float(data["classification_report"]["macro avg"]["f1-score"])
    return acc, macro_f1


def _gather_data(ks: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params_list = []
    acc_list = []
    f1_list = []
    for k in ks:
        info = describe_model(k=k)
        params_list.append(info["trainable_params"])
        acc, macro_f1 = _load_metrics_for_k(k)
        acc_list.append(acc)
        f1_list.append(macro_f1)
    return np.array(params_list, dtype=np.float64), np.array(acc_list), np.array(f1_list)


def _smooth_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = min(3, len(x) - 1)
    coeffs = np.polyfit(np.log10(x), y, deg=order)
    xs = np.linspace(np.log10(x.min()), np.log10(x.max()), 256)
    ys = np.polyval(coeffs, xs)
    return xs, ys


def run(
    ks: List[int] | None = None,
    output_path: str | None = None,
) -> Path:
    if ks is None:
        ks = [1, 2, 4, 8, 16, 32, 64]

    params, acc, f1 = _gather_data(ks)

    repo_root = Path(__file__).resolve().parents[1]
    figures_dir = repo_root / ".cache" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else figures_dir / "scaledcnn_capacity_vs_performance.pdf"

    x_log = np.log10(params)
    acc_xs, acc_ys = _smooth_curve(params, acc)
    f1_xs, f1_ys = _smooth_curve(params, f1)

    plt.figure(figsize=(6, 4))
    plt.scatter(x_log, acc, color="#1f77b4", label="Accuracy", marker="o")
    plt.scatter(x_log, f1, color="#ff7f0e", label="Macro F1", marker="s")
    plt.plot(acc_xs, acc_ys, color="#1f77b4", linestyle=":", linewidth=2)
    plt.plot(f1_xs, f1_ys, color="#ff7f0e", linestyle=":", linewidth=2)

    xticks = np.log10(params)
    xtick_labels = [f"{p/1e6:.2f}M" for p in params]
    plt.xticks(xticks, xtick_labels, rotation=45)

    plt.xlabel("Trainable Parameters")
    plt.ylabel("Score")
    plt.ylim(0.6, 0.8)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(output), format="pdf")
    plt.close()

    print(f"Saved capacity vs performance plot to: {output}")
    return output


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "capacity-curve",
        help="Plot accuracy and macro F1 vs parameter count",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output path for the capacity curve PDF.",
    )

    def _entry(args):
        return run(output_path=args.output_path)

    parser.set_defaults(entry=_entry)
    return parser


