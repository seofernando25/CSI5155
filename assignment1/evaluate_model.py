#!/usr/bin/env python3
"""
Model evaluation script.
Evaluates the trained model on the held-out test set for unbiased performance assessment.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load the held-out test data."""
    print("Loading test data...")
    
    # Load features and target
    X_test = pd.read_csv('data/test_features.csv')
    y_test = pd.read_csv('data/test_target.csv').squeeze()
    
    print(f"Test samples: {len(X_test)}")
    print(f"Target ratio: {y_test.mean():.4f}")
    
    return X_test, y_test

def load_trained_model():
    """Load the trained model and model info."""
    print("Loading trained model...")
    
    # Load model
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load model info
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print(f"Model: {model_info['model_name']}")
    print(f"Sampling: {model_info['sampling_strategy']}")
    
    return model, model_info

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    print("\nEvaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"Test Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }

def plot_confusion_matrix(cm, model_name, sampling_strategy):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Complain', 'Complain'],
                yticklabels=['No Complain', 'Complain'])
    plt.title(f'Confusion Matrix - {model_name.upper()} ({sampling_strategy})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('figures/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, model_name, sampling_strategy):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.upper()} ({sampling_strategy})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/test_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(test_results, model_info):
    """Save evaluation results."""
    print("\nSaving evaluation results...")
    
    # Combine model info and test results
    final_results = {
        'model_info': model_info,
        'test_results': test_results,
        'evaluation_summary': {
            'model': model_info['model_name'],
            'sampling': model_info['sampling_strategy'],
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_specificity': test_results['specificity'],
            'test_roc_auc': test_results['roc_auc']
        }
    }
    
    # Save results
    with open('test_evaluation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("Results saved to: test_evaluation_results.json")

def print_summary(test_results, model_info):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_info['model_name'].upper()}")
    print(f"Sampling Strategy: {model_info['sampling_strategy']}")
    print(f"Best CV Score (from tuning): {model_info['best_cv_score']:.4f}")
    print(f"Training Recall: {model_info['training_metrics']['recall']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print("\nTest Set Performance:")
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  Precision:   {test_results['precision']:.4f}")
    print(f"  Recall:      {test_results['recall']:.4f}")
    print(f"  Specificity: {test_results['specificity']:.4f}")
    print(f"  ROC AUC:     {test_results['roc_auc']:.4f}")
    print("="*60)

def main():
    """Main evaluation function."""
    print("="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load trained model
    model, model_info = load_trained_model()
    
    # Evaluate model
    test_results = evaluate_model(model, X_test, y_test)
    
    # Create plots
    print("\nCreating evaluation plots...")
    plot_confusion_matrix(test_results['confusion_matrix'], 
                         model_info['model_name'], 
                         model_info['sampling_strategy'])
    plot_roc_curve(y_test, test_results['probabilities'], 
                   model_info['model_name'], 
                   model_info['sampling_strategy'])
    
    # Save results
    save_results(test_results, model_info)
    
    # Print summary
    print_summary(test_results, model_info)
    
    print("\nEvaluation completed!")
    print("Check figures/test_confusion_matrix.png and figures/test_roc_curve.png")

if __name__ == '__main__':
    main()
