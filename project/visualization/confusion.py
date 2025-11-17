from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    model_name: str = "Model",
    output_path: Path | str | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> Path:
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage"},
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if output_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / ".cache" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in model_name
        )
        output_path = output_dir / f"{safe_name}_confusion_matrix.pdf"
    else:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
    return output_path


