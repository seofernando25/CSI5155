from lib.data_utils import CACHE_DIR, load_results
from lib.plot_utils import plot_accuracy, plot_metric


def main():
    baseline_results = load_results(CACHE_DIR / "results" / "baseline.json")
    self_training_results = load_results(CACHE_DIR / "results" / "self_training.json")
    label_prop_results = load_results(CACHE_DIR / "results" / "label_propagation.json")

    plot_accuracy(
        baseline_results,
        self_training_results,
        label_prop_results,
        "accuracy",
        "Accuracy vs Label Proportion",
    )
    plot_metric(
        baseline_results,
        self_training_results,
        label_prop_results,
        "macro avg",
        "f1-score",
        "macro_f1",
        "Macro-F1 vs Label Proportion",
    )
    plot_metric(
        baseline_results,
        self_training_results,
        label_prop_results,
        "macro avg",
        "precision",
        "macro_precision",
        "Macro-Precision vs Label Proportion",
    )
    plot_metric(
        baseline_results,
        self_training_results,
        label_prop_results,
        "macro avg",
        "recall",
        "macro_recall",
        "Macro-Recall vs Label Proportion",
    )
    plot_metric(
        baseline_results,
        self_training_results,
        label_prop_results,
        "6",
        "f1-score",
        "f1_class_6",
        "F1 (class 6) vs Label Proportion",
    )
    plot_metric(
        baseline_results,
        self_training_results,
        label_prop_results,
        "8",
        "f1-score",
        "f1_class_8",
        "F1 (class 8) vs Label Proportion",
    )


if __name__ == "__main__":
    main()
