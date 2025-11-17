from scaledcnn.evaluation import summarize_classification_results
from data import CIFAR10_CLASS_NAMES, get_cifar10_split
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM
from utils import require_file


def main():
    # Load model
    model_path = require_file(SVM_CLASSIFIER_PATH, hint="Train the model first")

    model = ClassifierSVM.load(str(model_path))

    # Load datasets
    X_train, y_train = get_cifar10_split("train")
    X_test, y_test = get_cifar10_split("test")

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    # Summarize results using shared utility
    class_names = CIFAR10_CLASS_NAMES
    summarize_classification_results(
        labels=y_test,
        predictions=predictions,
        class_names=class_names,
    )

    print(
        f"\nTrain: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%) | Test: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
