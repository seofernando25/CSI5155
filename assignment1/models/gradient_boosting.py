"""Gradient Boosting model definition."""

from sklearn.ensemble import GradientBoostingClassifier


def get_model():
    """Get a Gradient Boosting model instance."""
    return GradientBoostingClassifier(random_state=42)


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 1.0],
        'clf__min_samples_split': [2, 10],
    }


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return False

