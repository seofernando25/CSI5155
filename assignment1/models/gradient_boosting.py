from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Real, Integer, Categorical


def get_model():
    return GradientBoostingClassifier()


def get_param_space():
    return {
        "clf__n_estimators": Integer(50, 150),
        "clf__learning_rate": Real(0.001, 1.0),
        "clf__max_depth": Integer(3, 10),
        "clf__subsample": Real(0.5, 1.0),
        "clf__min_samples_split": Integer(5, 20),
        "clf__min_samples_leaf": Integer(2, 10),
        "clf__max_features": Categorical([None, "sqrt", "log2"]),
        "clf__loss": Categorical(["log_loss", "exponential"]),
    }
