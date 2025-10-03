"""k-Nearest Neighbors model definition."""

from sklearn.neighbors import KNeighborsClassifier


def get_model():
    """Get a k-NN model instance."""
    return KNeighborsClassifier()


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return {
        'clf__n_neighbors': [3, 5, 7, 11, 15, 21],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan'],
    }


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return True

