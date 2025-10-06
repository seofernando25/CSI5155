import os
import json
import argparse
import time
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from sklearn.dummy import ClassifierMixin
from skopt.utils import Dimension
from tqdm import tqdm

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.metrics import specificity_score
from skopt import BayesSearchCV
from pipeline import (
    load_dataframe,
    split_features,
    build_preprocessor,
    build_pipeline,
)
from sampling import SAMPLING_STRATEGIES
from models import MODELS, get_model_module

CPU_COUNT = os.cpu_count()

N_ITER = 5


def tune_model(
    model_name: str,
    get_model_fn: Callable[[], ClassifierMixin],
    get_param_space_fn: Callable[[], dict[str, Dimension]],
    sampling: str = "none",
    output_file: str = "tuning_results.json",
):
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)

    preprocessor = build_preprocessor(num_cols, cat_cols)

    clf = get_model_fn()
    param_space = get_param_space_fn()

    estimator = build_pipeline(preprocessor, clf, sampling=sampling)

    recall_scorer = make_scorer(recall_score)
    grid = BayesSearchCV(
        estimator,
        search_spaces=param_space,
        n_iter=N_ITER,
        n_jobs=-1,
        verbose=1,
        scoring=recall_scorer,
    )

    print(
        f"[{model_name}][{sampling}] Starting Bayesian optimization (5 iterations) - optimizing for recall (catching complainers)..."
    )

    start_time = time.time()
    iteration_count = [0]

    def progress_callback(_):
        iteration_count[0] += 1
        elapsed = time.time() - start_time
        print(
            f"[{model_name}][{sampling}] Iteration {iteration_count[0]}/5 - Elapsed: {elapsed:.1f}s"
        )

    grid.fit(X, y, callback=progress_callback)
    best_est = grid.best_estimator_

    total_time = time.time() - start_time
    print(
        f"[{model_name}][{sampling}] Completed in {total_time:.1f}s - Best params: {grid.best_params_}"
    )

    probas = cross_val_predict(best_est, X, y, cv=2, method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    roc_auc = roc_auc_score(y, probas)

    y_pred = (probas >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    specificity = specificity_score(y, y_pred)

    results = {
        "model": model_name,
        "sampling": sampling,
        "best_params": grid.best_params_,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
        },
    }

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    updated = False
    for i, r in enumerate(all_results):
        if r["model"] == model_name and r["sampling"] == sampling:
            all_results[i] = results
            updated = True
            break

    if not updated:
        all_results.append(results)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"[{model_name}][{sampling}] precision={precision:.4f} recall={recall:.4f} specificity={specificity:.4f} auc={roc_auc:.4f}"
    )

    return results


def run_tuning_job(model: str, sampling: str):
    cmd = ["uv", "run", "python", "tuning.py", "--model", model, "--sampling", sampling]

    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0

    if not ok:
        message = result.stderr.strip() or result.stdout.strip()
    else:
        message = "OK"

    return f"{model}[{sampling}]", message, result.returncode


def run_all_tuning(max_workers: int = CPU_COUNT):
    jobs = list(itertools.product(MODELS, SAMPLING_STRATEGIES))

    print(f"Running {len(jobs)} tuning jobs with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for model, sampling in jobs:
            future = executor.submit(run_tuning_job, model, sampling)
            futures[future] = (model, sampling)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tuning"):
            job_id, message, code = future.result()

            if code != 0:
                print(f"\n[ERROR] {job_id}:")
                print(message)
            else:
                print(f"\n[OK] {job_id}")
                if message:
                    print(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=MODELS
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_STRATEGIES,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CPU_COUNT,
    )
    args = parser.parse_args()

    if args.model and args.sampling:
        model_module = get_model_module(args.model)

        tune_model(
            args.model,
            model_module.get_model,
            model_module.get_param_space,
            sampling=args.sampling,
        )

    else:
        run_all_tuning(max_workers=args.workers)


if __name__ == "__main__":
    main()
