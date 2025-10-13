#!/usr/bin/env python3
"""
Comprehensive evaluation script for all model-sampling combinations.
Generates metrics needed for figure generation including confusion matrices and ROC curves.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from pipeline import get_pipeline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all data splits."""
    print("Loading data splits...")
    
    # Load features
    X_train = pd.read_csv('data/train_features.csv')
    X_val = pd.read_csv('data/val_features.csv')
    X_test = pd.read_csv('data/test_features.csv')
    
    # Load targets
    y_train = pd.read_csv('data/train_target.csv').squeeze()
    y_val = pd.read_csv('data/val_target.csv').squeeze()
    y_test = pd.read_csv('data/test_target.csv').squeeze()
    
    # Combine train and val for evaluation
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    print(f"Train+Val samples: {len(X_train_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train_val, y_train_val, X_test, y_test

def load_tuning_results():
    """Load the tuning results to get best parameters."""
    with open('tuning_results.json', 'r') as f:
        return json.load(f)

def evaluate_model_comprehensive(model_name, sampling_strategy, best_params, X_train_val, y_train_val, X_test, y_test):
    """Evaluate a model comprehensively with both 2-fold and 5-fold CV."""
    print(f"Evaluating {model_name} with {sampling_strategy} sampling...")
    
    # Get pipeline
    pipeline = get_pipeline(model_name, sampling_strategy, X_train_val)
    pipeline.set_params(**best_params)
    
    # 2-fold CV evaluation (for consistency with tuning)
    print(f"  Running 2-fold CV...")
    cv2_scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=2, scoring='recall')
    cv2_predictions = cross_val_predict(pipeline, X_train_val, y_train_val, cv=2, method='predict')
    cv2_probabilities = cross_val_predict(pipeline, X_train_val, y_train_val, cv=2, method='predict_proba')[:, 1]
    
    # 5-fold CV evaluation
    print(f"  Running 5-fold CV...")
    cv5_scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=5, scoring='recall')
    cv5_predictions = cross_val_predict(pipeline, X_train_val, y_train_val, cv=5, method='predict')
    cv5_probabilities = cross_val_predict(pipeline, X_train_val, y_train_val, cv=5, method='predict_proba')[:, 1]
    
    # Calculate metrics for 2-fold CV
    cv2_precision = precision_score(y_train_val, cv2_predictions, zero_division=0)
    cv2_recall = recall_score(y_train_val, cv2_predictions, zero_division=0)
    cv2_roc_auc = roc_auc_score(y_train_val, cv2_probabilities)
    cv2_cm = confusion_matrix(y_train_val, cv2_predictions)
    
    # Calculate metrics for 5-fold CV
    cv5_precision = precision_score(y_train_val, cv5_predictions, zero_division=0)
    cv5_recall = recall_score(y_train_val, cv5_predictions, zero_division=0)
    cv5_roc_auc = roc_auc_score(y_train_val, cv5_probabilities)
    cv5_cm = confusion_matrix(y_train_val, cv5_predictions)
    
    # Calculate specificity for both
    tn2, fp2, fn2, tp2 = cv2_cm.ravel()
    tn5, fp5, fn5, tp5 = cv5_cm.ravel()
    cv2_specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
    cv5_specificity = tn5 / (tn5 + fp5) if (tn5 + fp5) > 0 else 0
    
    return {
        'model': model_name,
        'sampling': sampling_strategy,
        'best_params': best_params,
        'cv2_metrics': {
            'precision': cv2_precision,
            'recall': cv2_recall,
            'specificity': cv2_specificity,
            'roc_auc': cv2_roc_auc,
            'confusion_matrix': cv2_cm.tolist(),
            'predictions': cv2_predictions.tolist(),
            'probabilities': cv2_probabilities.tolist()
        },
        'cv5_metrics': {
            'precision': cv5_precision,
            'recall': cv5_recall,
            'specificity': cv5_specificity,
            'roc_auc': cv5_roc_auc,
            'confusion_matrix': cv5_cm.tolist(),
            'predictions': cv5_predictions.tolist(),
            'probabilities': cv5_probabilities.tolist()
        }
    }

def main():
    """Main evaluation function."""
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Load data
    X_train_val, y_train_val, X_test, y_test = load_data()
    
    # Load tuning results
    tuning_results = load_tuning_results()
    
    # Evaluate all model-sampling combinations
    all_results = []
    
    for result in tuning_results:
        model_name = result['model']
        sampling_strategy = result['sampling']
        best_params = result['best_params']
        
        try:
            eval_result = evaluate_model_comprehensive(
                model_name, sampling_strategy, best_params,
                X_train_val, y_train_val, X_test, y_test
            )
            all_results.append(eval_result)
        except Exception as e:
            print(f"Error evaluating {model_name}[{sampling_strategy}]: {e}")
            continue
    
    # Save comprehensive results
    with open('comprehensive_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComprehensive evaluation completed!")
    print(f"Results saved to: comprehensive_evaluation_results.json")
    print(f"Total combinations evaluated: {len(all_results)}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("2-Fold CV Results:")
    for result in all_results:
        metrics = result['cv2_metrics']
        print(f"{result['model']}[{result['sampling']}]: "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
              f"AUC={metrics['roc_auc']:.4f}")
    
    print("\n5-Fold CV Results:")
    for result in all_results:
        metrics = result['cv5_metrics']
        print(f"{result['model']}[{result['sampling']}]: "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
              f"AUC={metrics['roc_auc']:.4f}")

if __name__ == '__main__':
    main()



