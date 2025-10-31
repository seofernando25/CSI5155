from __future__ import annotations
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from lib.semi_supervised_utils import run_semi_supervised_experiments


def create_self_training_model():
    """Create a self-training classifier model."""
    base = SVC(C=0.01, kernel="linear", probability=True)
    return SelfTrainingClassifier(base, criterion="k_best", max_iter=100)


def main() -> None:
    run_semi_supervised_experiments(
        create_semi_supervised_model=create_self_training_model,
        results_filename="self_training.json",
        n_runs=5,
    )


if __name__ == "__main__":
    main()
