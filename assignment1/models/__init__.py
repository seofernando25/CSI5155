from . import (
    logistic_regression,
    decision_tree,
    svm,
    knn,
    random_forest,
    gradient_boosting,
)

MODELS = ("lr", "dt", "svm", "knn", "rf", "gb")

MODEL_MODULES = {
    "lr": logistic_regression,
    "dt": decision_tree,
    "svm": svm,
    "knn": knn,
    "rf": random_forest,
    "gb": gradient_boosting,
}

MODEL_LABELS = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "svm": "SVM",
    "knn": "k-NN",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

def get_model_module(code: str):
    return MODEL_MODULES[code]



