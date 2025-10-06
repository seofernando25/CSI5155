from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer, Categorical


def get_model():
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1)


def get_param_space():
    return {
        "clf__n_estimators": Integer(50, 150),
        "clf__max_depth": Integer(5, 20),
        "clf__min_samples_split": Integer(5, 20),
        "clf__min_samples_leaf": Integer(2, 10),
        "clf__max_features": Categorical(["sqrt", "log2", None]),
        "clf__bootstrap": Categorical([True, False]),
        "clf__min_impurity_decrease": Real(0.0, 0.1),
    }
