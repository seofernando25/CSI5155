"""Decision Tree model definition."""

from sklearn.tree import DecisionTreeClassifier


def get_model():
    """Get a Decision Tree model instance."""
    return DecisionTreeClassifier(
        random_state=42
    )


def get_param_grid():
    """Get the hyperparameter grid for tuning."""
    return {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [5, 10, 15, 20, None],
        'clf__min_samples_split': [2, 10, 20],
        'clf__min_samples_leaf': [1, 2],
        'clf__max_features': [None, 'sqrt', 'log2'],
        'clf__class_weight': [None, 'balanced'],
    }


def needs_scaling():
    """Returns whether this model requires feature scaling."""
    return False

