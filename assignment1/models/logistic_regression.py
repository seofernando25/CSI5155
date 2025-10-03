"""Logistic Regression model definition."""

from sklearn.linear_model import LogisticRegression


def get_model():
    """Get a Logistic Regression model instance."""
    return LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        random_state=42
    )


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return {
        'clf__C': [0.005, 0.01, 0.05],
        'clf__penalty': ['l1', 'l2'],
        'clf__max_iter': [500, 1000, 1500],
    }


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return True

