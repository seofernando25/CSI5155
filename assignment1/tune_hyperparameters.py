#!/usr/bin/env python3
"""
Hyperparameter tuning script using Bayesian optimization.
Uses train+validation data with cross-validation to find best hyperparameters.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from pipeline import get_pipeline
import warnings
warnings.filterwarnings('ignore')

def load_train_val_data():
    """Load training and validation data."""
    print("Loading train and validation data...")
    
    # Load features
    X_train = pd.read_csv('data/train_features.csv')
    X_val = pd.read_csv('data/val_features.csv')
    
    # Load targets
    y_train = pd.read_csv('data/train_target.csv').squeeze()
    y_val = pd.read_csv('data/val_target.csv').squeeze()
    
    # Combine train and val for tuning
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    print(f"Train+Val samples: {len(X_train_val)}")
    print(f"Target ratio: {y_train_val.mean():.4f}")
    
    return X_train_val, y_train_val

def get_search_spaces():
    """Define hyperparameter search spaces for each model."""
    return {
        'lr': {
            'clf__C': Real(0.01, 100, prior='log-uniform'),
            'clf__max_iter': Integer(100, 1000),
            'clf__solver': Categorical(['liblinear', 'lbfgs', 'saga'])
        },
        'dt': {
            'clf__max_depth': Integer(3, 20),
            'clf__min_samples_split': Integer(2, 20),
            'clf__min_samples_leaf': Integer(1, 10),
            'clf__criterion': Categorical(['gini', 'entropy'])
        },
        'svm': {
            'clf__C': Real(0.01, 100, prior='log-uniform'),
            'clf__gamma': Real(0.001, 1, prior='log-uniform'),
            'clf__kernel': Categorical(['rbf', 'linear', 'poly'])
        },
        'knn': {
            'clf__n_neighbors': Integer(3, 20),
            'clf__weights': Categorical(['uniform', 'distance']),
            'clf__metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        },
        'rf': {
            'clf__n_estimators': Integer(50, 150),
            'clf__max_depth': Integer(3, 15),
            'clf__min_samples_split': Integer(2, 10),
            'clf__min_samples_leaf': Integer(1, 5)
        },
        'gb': {
            'clf__n_estimators': Integer(50, 150),
            'clf__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'clf__max_depth': Integer(3, 8),
            'clf__subsample': Real(0.7, 1.0)
        }
    }

def tune_model(model_name, sampling_strategy, X_train_val, y_train_val):
    """Tune hyperparameters for a specific model and sampling strategy."""
    print(f"\nTuning {model_name} with {sampling_strategy} sampling...")
    
    # Get pipeline with data to properly configure column transformers
    pipeline = get_pipeline(model_name, sampling_strategy, X_train_val)
    
    # Get search space
    search_spaces = get_search_spaces()
    search_space = search_spaces[model_name]
    
    # Bayesian optimization
    search = BayesSearchCV(
        pipeline,
        search_space,
        n_iter=10,  # Reduced from 50 to 10 iterations
        cv=3,       # Reduced from 5 to 3-fold cross-validation
        scoring='recall',  # Optimize for recall
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the search
    search.fit(X_train_val, y_train_val)
    
    # Get best parameters
    best_params = search.best_params_
    best_score = search.best_score_
    
    # Convert parameters to JSON-serializable format
    best_params_serializable = {}
    for key, value in best_params.items():
        if hasattr(value, 'item'):  # numpy scalar
            best_params_serializable[key] = value.item()
        else:
            best_params_serializable[key] = value
    
    print(f"Best CV recall: {best_score:.4f}")
    print(f"Best parameters: {best_params_serializable}")
    
    return {
        'model': model_name,
        'sampling': sampling_strategy,
        'best_params': best_params_serializable,
        'best_cv_score': best_score,
        'search_space': str(search_space)  # Convert to string for JSON
    }

def main():
    """Main tuning function."""
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    X_train_val, y_train_val = load_train_val_data()
    
    # Define models and sampling strategies
    models = ['lr', 'dt', 'svm', 'knn', 'rf', 'gb']
    sampling_strategies = ['none', 'undersample', 'smote']
    
    # Store results
    tuning_results = []
    
    # Tune each model with each sampling strategy
    for model in models:
        for sampling in sampling_strategies:
            try:
                result = tune_model(model, sampling, X_train_val, y_train_val)
                tuning_results.append(result)
            except Exception as e:
                print(f"Error tuning {model} with {sampling}: {e}")
                continue
    
    # Save results
    with open('tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\nTuning completed! Results saved to tuning_results.json")
    print(f"Total combinations tuned: {len(tuning_results)}")
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for result in tuning_results:
        print(f"{result['model']}[{result['sampling']}]: "
              f"CV recall={result['best_cv_score']:.4f}")

if __name__ == '__main__':
    main()
