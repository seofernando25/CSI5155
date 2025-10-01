#!/usr/bin/env python3
import os
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from models.utils import (
    load_dataframe,
    split_features,
    build_preprocessor,
    build_pipeline,
    evaluate_and_report,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'rf')


def run(sampling: str = 'none'):
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)

    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    clf = RandomForestClassifier(class_weight='balanced', n_estimators=400, random_state=42, n_jobs=-1)

    estimator = build_pipeline(pre, clf, sampling=sampling)

    param_grid = {
        'clf__max_depth': [None, 20],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 4],
    }

    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring={'recall': make_scorer(recall_score)},
        refit='recall',
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X, y)
    best_est = grid.best_estimator_

    print('Best params:', grid.best_params_)

    tag = f"RandomForest[{sampling}]"
    out_dir = os.path.join(OUT_DIR, sampling)
    metrics = evaluate_and_report(tag, best_est, X, y, out_dir)
    with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
        json.dump(grid.best_params_, f, indent=2)
    return metrics


if __name__ == '__main__':
    from models.utils import SAMPLING_STRATEGIES

    ap = argparse.ArgumentParser()
    ap.add_argument('--sampling', choices=SAMPLING_STRATEGIES, default='none')
    args = ap.parse_args()
    run(args.sampling)
