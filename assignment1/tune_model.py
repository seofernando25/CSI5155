#!/usr/bin/env python3
"""
Generic model tuning script.

This script provides a reusable function for tuning any model with grid search.
"""

import os
import json
import argparse
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, recall_score, precision_score, roc_auc_score, confusion_matrix

from preprocessing import load_dataframe, split_features, build_preprocessor
from utils import build_pipeline, SAMPLING_STRATEGIES


def tune_model(model_name: str, 
               get_model_fn, 
               get_param_grid_fn, 
               needs_scaling_fn,
               sampling: str = 'none',
               output_file: str = 'tuning_results.json'):
    """
    Tune a model using grid search and save results.
    
    Args:
        model_name: Name of the model (e.g., 'lr', 'dt')
        get_model_fn: Function that returns a model instance
        get_param_grid_fn: Function that returns parameter grid
        needs_scaling_fn: Function that returns whether scaling is needed
        sampling: Sampling strategy to use
        output_file: JSON file to save results
    """
    # Load data
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=needs_scaling_fn())
    
    # Get model and parameter grid
    clf = get_model_fn()
    param_grid = get_param_grid_fn()
    
    # Build full pipeline
    estimator = build_pipeline(preprocessor, clf, sampling=sampling)
    
    # Grid search with recall scoring
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    
    grid.fit(X, y)
    best_est = grid.best_estimator_
    
    print(f'\n[{model_name}] Best params: {grid.best_params_}')
    
    # Get cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probas = cross_val_predict(best_est, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    y_pred = (probas >= 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, probas)
    
    # Prepare results
    results = {
        'model': model_name,
        'sampling': sampling,
        'best_params': grid.best_params_,
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
    }
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Update or append result
    updated = False
    for i, r in enumerate(all_results):
        if r['model'] == model_name and r['sampling'] == sampling:
            all_results[i] = results
            updated = True
            break
    
    if not updated:
        all_results.append(results)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'[{model_name}][{sampling}] precision={precision:.4f} recall={recall:.4f} auc={roc_auc:.4f}')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                       choices=['lr', 'dt', 'svm', 'knn', 'rf', 'gb'])
    parser.add_argument('--sampling', choices=SAMPLING_STRATEGIES, default='none')
    args = parser.parse_args()
    
    # Import the appropriate model module
    if args.model == 'lr':
        from models import logistic_regression as model_module
    elif args.model == 'dt':
        from models import decision_tree as model_module
    elif args.model == 'svm':
        from models import svm as model_module
    elif args.model == 'knn':
        from models import knn as model_module
    elif args.model == 'rf':
        from models import random_forest as model_module
    elif args.model == 'gb':
        from models import gradient_boosting as model_module
    
    tune_model(
        args.model,
        model_module.get_model,
        model_module.get_param_grid,
        model_module.needs_scaling,
        sampling=args.sampling
    )

