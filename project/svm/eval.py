from pathlib import Path

from scaledcnn.evaluation import summarize_classification_results
from data import get_cifar10_class_names, get_cifar10_split
from svm.constants import SVM_CLASSIFIER_PATH
from svm.model import ClassifierSVM


def main():
    # Load model
    model_path = Path(SVM_CLASSIFIER_PATH)
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train the model first: uv run python -m svm.train")
        return

    model = ClassifierSVM.load(str(model_path))

    # Load datasets
    try:
        X_train, y_train = get_cifar10_split("train")
        X_test, y_test = get_cifar10_split("test")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    # Summarize results using shared utility
    class_names = get_cifar10_class_names()
    summarize_classification_results(
        labels=y_test,
        predictions=predictions,
        class_names=class_names,
    )
    
    print(f"\nTrain: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%) | Test: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
