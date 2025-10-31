from __future__ import annotations
from sklearn.semi_supervised import LabelPropagation
from lib.semi_supervised_utils import run_semi_supervised_experiments


def create_label_propagation_model():
    """Create a label propagation model."""
    return LabelPropagation(
        kernel="knn",
        n_neighbors=7,
    )


def main() -> None:
    run_semi_supervised_experiments(
        create_semi_supervised_model=create_label_propagation_model,
        results_filename="label_propagation.json",
        n_runs=5,
    )


if __name__ == "__main__":
    main()
