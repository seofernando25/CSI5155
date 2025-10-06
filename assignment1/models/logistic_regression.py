from sklearn.linear_model import LogisticRegression
from skopt.space import Real, Integer, Categorical


def get_model():
    return LogisticRegression(
        class_weight="balanced",
        solver="saga",  # needed for elasticnet because why not...
    )


def get_param_space():
    return {
        "clf__C": Real(0.001, 10.0),
        "clf__penalty": Categorical(["l1", "l2", "elasticnet"]),
        "clf__max_iter": Integer(1000, 5000),
        "clf__l1_ratio": Real(0.0, 1.0),
        "clf__class_weight": Categorical([None, "balanced"]),
    }
