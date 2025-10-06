from sklearn.tree import DecisionTreeClassifier
from skopt.space import Integer, Categorical


def get_model():
    return DecisionTreeClassifier()


def get_param_space():
    return {
        "clf__criterion": Categorical(["gini", "entropy"]),
        "clf__max_depth": Integer(3, 15),
        "clf__min_samples_split": Integer(5, 20),
        "clf__min_samples_leaf": Integer(2, 10),
        "clf__max_features": Categorical([None, "sqrt", "log2"]),
        "clf__class_weight": Categorical([None, "balanced"]),
    }
