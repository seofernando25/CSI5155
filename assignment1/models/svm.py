from sklearn.svm import SVC
from skopt.space import Real, Categorical


def get_model():
    return SVC(
        class_weight="balanced",
        probability=True,
    )


def get_param_space():
    return {
        "clf__kernel": Categorical(["linear", "rbf"]),
        "clf__C": Real(0.001, 100.0),
        "clf__gamma": Real(0.0001, 10.0),
        "clf__class_weight": Categorical([None, "balanced"]),
    }
