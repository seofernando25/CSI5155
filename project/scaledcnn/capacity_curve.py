from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from paths import FIGURES_DIR, METRICS_DIR
from scaledcnn.info import describe_model


def _interpolate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    # Gather data for all k values
    params_list = []
    acc_list = []
    f1_list = []
    for k in ks:
        info = describe_model(k=k)
        params_list.append(info["trainable_params"])
        # Load metrics for this k value
        metrics_path = METRICS_DIR / f"scaledcnn_k{k}_test_training_report.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics JSON not found at {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        acc = float(data["classification_report"]["accuracy"])
        macro_f1 = float(data["classification_report"]["macro avg"]["f1-score"])
        acc_list.append(acc)
        f1_list.append(macro_f1)
    params = np.array(params_list, dtype=np.float64)
    acc = np.array(acc_list)
    f1 = np.array(f1_list)
    output = Path(output_path) if output_path else FIGURES_DIR / "scaledcnn_capacity_vs_performance.pdf"

    x_log = np.log10(params)
    acc_xs, acc_ys = _interpolate(params, acc)
    f1_xs, f1_ys = _interpolate(params, f1)

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
    parser.set_defaults(entry=lambda args: run(output_path=args.output_path))
    return parser


