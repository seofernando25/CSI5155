"""Random Forest model definition."""

from sklearn.ensemble import RandomForestClassifier


def get_model():
    """Get a Random Forest model instance."""
    return RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [10, 20, 30, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', None],
    }


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return False

