#!/usr/bin/env python3
"""Generate combined ROC curves for all models."""

import os
import json
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing import load_dataframe, split_features, build_preprocessor
from utils import build_pipeline


# Model configurations
MODELS = {
    'lr': ('Logistic Regression', 'logistic_regression'),
    'dt': ('Decision Tree', 'decision_tree'),
    'svm': ('SVM', 'svm'),
    'knn': ('k-NN', 'knn'),
    'rf': ('Random Forest', 'random_forest'),
    'gb': ('Gradient Boosting', 'gradient_boosting'),
}

SAMPLINGS = {
    'none': 'No Sampling (Baseline)',
    'under': 'Random Undersampling',
    'smote': 'SMOTE',
}

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


def get_model_instance(model_key, best_params):
    """Create a model instance with best parameters."""
    from models import (
        logistic_regression, decision_tree, svm, 
        knn, random_forest, gradient_boosting
    )
    
    module_map = {
        'lr': logistic_regression,
        'dt': decision_tree,
        'svm': svm,
        'knn': knn,
        'rf': random_forest,
        'gb': gradient_boosting,
    }
    
    module = module_map[model_key]
    clf = module.get_model()
    
    # Set parameters (strip 'clf__' prefix)
    params = {k.replace('clf__', ''): v for k, v in best_params.items()}
    clf.set_params(**params)
    
    return clf, module.needs_scaling()


def compute_roc_data(model_key, sampling, X, y, num_cols, cat_cols, tuning_results):
    """Compute ROC curve data for a model."""
    # Find the result for this model/sampling combination
    best_params = None
    for result in tuning_results:
        if result['model'] == model_key and result['sampling'] == sampling:
            best_params = result['best_params']
            break
    
    if best_params is None:
        print(f"Warning: No tuning results found for {model_key}/{sampling}, skipping...")
        return None, None, None
    
    # Create model with best parameters
    clf, needs_scaling = get_model_instance(model_key, best_params)
    
    # Build pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=needs_scaling)
    estimator = build_pipeline(preprocessor, clf, sampling=sampling)
    
    # Get out-of-fold predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probas = cross_val_predict(estimator, X, y, cv=cv, 
                               method='predict_proba', n_jobs=-1)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y, probas)
    auc = roc_auc_score(y, probas)
    
    return fpr, tpr, auc


def generate_combined_roc_curves(output_dir='figures', tuning_file='tuning_results.json'):
    """Generate combined ROC curve plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tuning results
    try:
        with open(tuning_file, 'r') as f:
            tuning_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {tuning_file} not found. Run tuning first.")
        return
    
    # Load data
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)
    
    # Generate 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (sampling_key, sampling_label) in enumerate(SAMPLINGS.items()):
        ax = axes[idx]
        print(f"\nProcessing {sampling_label}...")
        
        for model_idx, (model_key, (model_label, _)) in enumerate(MODELS.items()):
            fpr, tpr, auc = compute_roc_data(model_key, sampling_key, X, y, num_cols, cat_cols, tuning_results)
            
            if fpr is not None:
                ax.plot(fpr, tpr, color=COLORS[model_idx], lw=2,
                       label=f'{model_label} (AUC={auc:.3f})')
                print(f"  {model_label}: AUC={auc:.3f}")
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curves - {sampling_label}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_all_models.png'), 
               dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/roc_curves_all_models.png")
    plt.close()


def generate_individual_roc_curves(output_dir='figures', tuning_file='tuning_results.json'):
    """Generate individual ROC curve plots for each sampling strategy."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tuning results
    try:
        with open(tuning_file, 'r') as f:
            tuning_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {tuning_file} not found. Run tuning first.")
        return
    
    # Load data
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)
    
    for sampling_key, sampling_label in SAMPLINGS.items():
        plt.figure(figsize=(10, 8))
        print(f"\nGenerating plot for {sampling_label}...")
        
        for model_idx, (model_key, (model_label, _)) in enumerate(MODELS.items()):
            fpr, tpr, auc = compute_roc_data(model_key, sampling_key, X, y, num_cols, cat_cols, tuning_results)
            
            if fpr is not None:
                plt.plot(fpr, tpr, color=COLORS[model_idx], lw=2.5,
                        label=f'{model_label} (AUC={auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title(f'ROC Curves - {sampling_label}', fontsize=15, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'roc_curves_{sampling_key}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


if __name__ == '__main__':
    print("Generating ROC curves...")
    generate_combined_roc_curves()
    generate_individual_roc_curves()
    print("\nAll ROC curve figures generated successfully!")

