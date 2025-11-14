import time
from pathlib import Path

from common_net import summarize_classification_results
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

    print(f"Loading model from: {model_path}")
    model = ClassifierSVM.load(str(model_path))
    print("Model loaded successfully!")

    # Load test dataset
    print("\nLoading CIFAR-10 test dataset...")
    try:
        X_test, y_test = get_cifar10_split("test")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return
    print(f"Test samples: {len(X_test)}")

    # Evaluate model
    print("\nEvaluating model on test set...")
    start_time = time.time()
    test_accuracy = model.score(X_test, y_test)
    eval_time = time.time() - start_time

    # Get predictions for detailed analysis
    print("Computing predictions...")
    start_time = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start_time

    # Summarize results using shared utility
    class_names = get_cifar10_class_names()
    metrics = summarize_classification_results(
        labels=y_test,
        predictions=predictions,
        class_names=class_names,
        evaluation_time=eval_time,
        prediction_time=predict_time,
    )
    metrics["accuracy"] = test_accuracy
    metrics["evaluation_time"] = eval_time
    metrics["prediction_time"] = predict_time


if __name__ == "__main__":
    main()
