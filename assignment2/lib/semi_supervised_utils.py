from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable
from sklearn.metrics import classification_report
from lib.data_utils import CACHE_DIR
from lib.data_utils import get_train_test_split, save_results

PROPORTIONS = [
    0.002,
    0.004,
    0.006,
    0.008,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.10,
]


def label_mask(y_train: pd.Series, p: float):
    n_labeled = max(1, int(len(y_train) * p))
    labeled_indices = np.random.choice(len(y_train), n_labeled, replace=False)
    y_masked = np.full(len(y_train), -1)
    y_masked[labeled_indices] = y_train.iloc[labeled_indices].values
    return y_masked


def train_models_for_p(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    p: float,
    create_semi_supervised_model: Callable,
) -> dict:
    y_train_masked = label_mask(y_train, p=p)

    model = create_semi_supervised_model()
    model.fit(X_train, y_train_masked)
    y_pred_semi = model.predict(X_test)

    return classification_report(y_test.to_numpy(), y_pred_semi, output_dict=True)


def run_semi_supervised_experiments(
    create_semi_supervised_model: Callable,
    results_filename: str,
    n_runs: int = 1,
):
    """Run semi-supervised experiments with the given model creator."""
    X_tr, X_te, y_tr, y_te = get_train_test_split()

    results = {}
    for p in PROPORTIONS:
        run_results = []
        for i in range(n_runs):
            metrics = train_models_for_p(
                X_tr,
                y_tr,
                X_te,
                y_te,
                p,
                create_semi_supervised_model=create_semi_supervised_model,
            )
            run_results.append(metrics)
            print(f"Completed run {i+1}/{n_runs} for p={p:.3%}")
        results[p] = run_results

    save_results(results, CACHE_DIR / "results" / results_filename)
    return results
