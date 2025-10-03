"""Support Vector Machine model definition."""

from sklearn.svm import SVC


def get_model():
    """Get an SVM model instance."""
    return SVC(
        class_weight='balanced',
        probability=True,
    )


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return [
        {
            'clf__kernel': ['linear'],
            'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            'clf__class_weight': [None, 'balanced'],
        },
        {
            'clf__kernel': ['rbf'],
            'clf__C': [0.01, 0.1, 1.0],
            'clf__gamma': ['scale', 'auto', 0.01],
            'clf__class_weight': [None, 'balanced'],
        },
    ]


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return True

