"""Common utilities for figure generation scripts."""

import os
import json
import matplotlib.pyplot as plt
from .theme_config import FIGURE_CONFIG, FONT_CONFIG, FIGURE_SIZES


def save_figure(filename, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, **FIGURE_CONFIG)
    print(f"Saved: {full_path}")
    plt.close()


def load_tuning_results(tuning_file="tuning_results.json"):
    try:
        with open(tuning_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {tuning_file} not found. Run tuning first.")
        return None


def find_metrics_for_model(tuning_results, model_key, sampling_key):
    for result in tuning_results:
        if result["model"] == model_key and result["sampling"] == sampling_key:
            return result["metrics"]
    return None


def setup_subplot_grid(nrows=2, ncols=3, figsize="medium"):
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGURE_SIZES[figsize])
    return fig, axes.flatten()


def apply_standard_plotting(ax, title, xlabel=None, ylabel=None, grid=True):
    ax.set_title(title, fontsize=FONT_CONFIG["label"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_CONFIG["tick"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_CONFIG["tick"])
    if grid:
        ax.grid(True)


def add_suptitle(fig, title):
    """Add a standard suptitle to the figure."""
    fig.suptitle(title, fontsize=FONT_CONFIG["title"], fontweight="bold", y=0.995)
    plt.tight_layout()
