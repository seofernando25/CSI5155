from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer, Categorical


def get_model():
    return KNeighborsClassifier()


def get_param_space():
    return {
        "clf__n_neighbors": Integer(1, 10),
        "clf__weights": Categorical(["uniform", "distance"]),
        "clf__metric": Categorical(["euclidean", "manhattan"]),
    }
