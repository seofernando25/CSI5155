#!/usr/bin/env python3
"""
Final model training script.
Trains the best model on all training data using optimized hyperparameters.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pipeline import get_pipeline
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_all_training_data():
    """Load all training data (train + validation combined)."""
    print("Loading all training data...")
    
    # Load features
    X_train = pd.read_csv('data/train_features.csv')
    X_val = pd.read_csv('data/val_features.csv')
    
    # Load targets
    y_train = pd.read_csv('data/train_target.csv').squeeze()
    y_val = pd.read_csv('data/val_target.csv').squeeze()
    
    # Combine train and val
    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)
    
    print(f"Total training samples: {len(X_all)}")
    print(f"Target ratio: {y_all.mean():.4f}")
    
    return X_all, y_all

def find_best_model():
    """Find the best model based on tuning results."""
    print("Loading tuning results...")
    
    with open('tuning_results.json', 'r') as f:
        tuning_results = json.load(f)
    
    # Find best model based on CV recall
    best_result = max(tuning_results, key=lambda x: x['best_cv_score'])
    
    print(f"Best model: {best_result['model']}")
    print(f"Best sampling: {best_result['sampling']}")
    print(f"Best CV recall: {best_result['best_cv_score']:.4f}")
    
    return best_result

def train_final_model(best_result, X_all, y_all):
    """Train the final model with best hyperparameters."""
    print(f"\nTraining final model: {best_result['model']} with {best_result['sampling']} sampling...")
    
    # Get pipeline with best parameters and data
    pipeline = get_pipeline(best_result['model'], best_result['sampling'], X_all)
    pipeline.set_params(**best_result['best_params'])
    
    # Train on all data
    pipeline.fit(X_all, y_all)
    
    # Evaluate on training data (for reference)
    y_pred = pipeline.predict(X_all)
    y_pred_proba = pipeline.predict_proba(X_all)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_all, y_pred)
    recall = recall_score(y_all, y_pred)
    roc_auc = roc_auc_score(y_all, y_pred_proba)
    cm = confusion_matrix(y_all, y_pred)
    
    print(f"Training metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return pipeline, {
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def save_model_and_results(pipeline, training_metrics, best_result):
    """Save the trained model and results."""
    print("\nSaving model and results...")
    
    # Save model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Save model info
    model_info = {
        'model_name': best_result['model'],
        'sampling_strategy': best_result['sampling'],
        'best_params': best_result['best_params'],
        'best_cv_score': best_result['best_cv_score'],
        'training_metrics': training_metrics
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Model saved to: best_model.pkl")
    print("Model info saved to: model_info.json")

def main():
    """Main training function."""
    print("="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    # Load all training data
    X_all, y_all = load_all_training_data()
    
    # Find best model from tuning
    best_result = find_best_model()
    
    # Train final model
    pipeline, training_metrics = train_final_model(best_result, X_all, y_all)
    
    # Save model and results
    save_model_and_results(pipeline, training_metrics, best_result)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print("Next step: Run evaluate_model.py to test on unseen data")

if __name__ == '__main__':
    main()
