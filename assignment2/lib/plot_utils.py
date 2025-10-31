import matplotlib.pyplot as plt
import numpy as np

from lib.data_utils import FIG_DIR
from lib.semi_supervised_utils import PROPORTIONS


def setup_plot(y_label: str):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Label proportion (p)")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    
    def logit_transform(x):
        x = np.clip(x, 0.001, 0.999)
        return np.log(x / (1 - x))

    def logit_inverse_transform(x):
        return 1 / (1 + np.exp(-x))

    plt.yscale("function", functions=(logit_transform, logit_inverse_transform))
    plt.xscale("log")

    ticks = np.linspace(-10, 10, 16)
    ticks = 1 / (1 + np.exp(-ticks))
    ticks = np.round(ticks, 2)
    ticks = np.unique(ticks)
    plt.yticks(ticks, [f"{y:.2f}" for y in ticks])

    # Set x-axis ticks to match our proportion values
    plt.xticks(PROPORTIONS, [f"{p:.1%}" for p in PROPORTIONS])


def save_plot(filename, title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename.name}")


def plot_metric(
    baseline_results,
    self_training_results,
    label_prop_results,
    metric_key: str,
    metric_name: str,
    filename: str,
    title: str,
):
    SEMI_SUPERVISED_FIG_DIR = FIG_DIR / "semi_supervised"
    SEMI_SUPERVISED_FIG_DIR.mkdir(parents=True, exist_ok=True)

    y_label = f"{metric_name.replace('-', ' ').title()}"
    if "class" in filename:
        y_label = f"Per-Class {y_label}"

    setup_plot(y_label=y_label)

    # Plot SVM baseline (constant line)
    baseline_metric = baseline_results["report"][metric_key][f"{metric_name}"]
    plt.plot(
        PROPORTIONS,
        [baseline_metric] * len(PROPORTIONS),
        marker="o",
        label="SVM (Supervised Baseline)",
        linestyle="--",
    )

    # Plot Self-Training
    self_training_scores = [
        [run[metric_key][f"{metric_name}"] for run in self_training_results[str(p)]]
        for p in PROPORTIONS
    ]
    self_training_mean = np.mean(self_training_scores, axis=1)
    self_training_std = np.std(self_training_scores, axis=1)

    plt.plot(
        PROPORTIONS, self_training_mean, marker="s", label="SVM (Self-Training)"
    )
    plt.fill_between(
        PROPORTIONS,
        self_training_mean - self_training_std,
        self_training_mean + self_training_std,
        alpha=0.2,
    )

    # Plot Label Propagation
    label_prop_scores = [
        [run[metric_key][f"{metric_name}"] for run in label_prop_results[str(p)]]
        for p in PROPORTIONS
    ]
    label_prop_mean = np.mean(label_prop_scores, axis=1)
    label_prop_std = np.std(label_prop_scores, axis=1)
    plt.plot(
        PROPORTIONS, label_prop_mean, marker="^", label="SVM (Label Propagation)"
    )
    plt.fill_between(
        PROPORTIONS,
        label_prop_mean - label_prop_std,
        label_prop_mean + label_prop_std,
        alpha=0.2,
    )

    plt.legend()
    save_plot(SEMI_SUPERVISED_FIG_DIR / f"{filename}.png", title)


def plot_accuracy(
    baseline_results,
    self_training_results,
    label_prop_results,
    filename: str,
    title: str,
):
    SEMI_SUPERVISED_FIG_DIR = FIG_DIR / "semi_supervised"
    SEMI_SUPERVISED_FIG_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot(y_label="Accuracy")

    # Plot SVM baseline (constant line)
    baseline_accuracy = baseline_results["report"]["accuracy"]
    plt.plot(
        PROPORTIONS,
        [baseline_accuracy] * len(PROPORTIONS),
        marker="o",
        label="SVM (Supervised Baseline)",
        linestyle="--",
    )

    # Plot Self-Training
    self_training_scores = [
        [run["accuracy"] for run in self_training_results[str(p)]]
        for p in PROPORTIONS
    ]
    self_training_mean = np.mean(self_training_scores, axis=1)
    self_training_std = np.std(self_training_scores, axis=1)

    plt.plot(
        PROPORTIONS, self_training_mean, marker="s", label="SVM (Self-Training)"
    )
    plt.fill_between(
        PROPORTIONS,
        self_training_mean - self_training_std,
        self_training_mean + self_training_std,
        alpha=0.2,
    )

    # Plot Label Propagation
    label_prop_scores = [
        [run["accuracy"] for run in label_prop_results[str(p)]] for p in PROPORTIONS
    ]
    label_prop_mean = np.mean(label_prop_scores, axis=1)
    label_prop_std = np.std(label_prop_scores, axis=1)
    plt.plot(
        PROPORTIONS, label_prop_mean, marker="^", label="SVM (Label Propagation)"
    )
    plt.fill_between(
        PROPORTIONS,
        label_prop_mean - label_prop_std,
        label_prop_mean + label_prop_std,
        alpha=0.2,
    )

    plt.legend()
    save_plot(SEMI_SUPERVISED_FIG_DIR / f"{filename}.png", title)
