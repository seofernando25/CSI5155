import time
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data import load_cifar10_data
from svm.model import ClassifierSVM
from svm.constants import SVM_CLASSIFIER_PATH


def run(model_path: str = SVM_CLASSIFIER_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/svm/benchmark/task_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir.resolve()}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        writer.close()
        raise FileNotFoundError(
            f"Model file not found at {model_path_obj}. Please train the model first: uv run python -m svm.train"
        )

    print(f"Loading trained model from: {model_path_obj}")
    model = ClassifierSVM.load(str(model_path_obj))
    print("Model loaded successfully!")

    print("\nLoading CIFAR-10 test dataset...")
    try:
        ds_dict = load_cifar10_data()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        writer.close()
        return None

    test_ds = ds_dict["test"]
    X_test = [np.asarray(item["img"], dtype=np.float32) for item in test_ds]
    y_test = np.array([item["label"] for item in test_ds])
    print(f"Test samples: {len(X_test)}")

    print("\nBenchmarking model on test set...")
    start_time = time.time()
    test_accuracy = model.score(X_test, y_test)
    eval_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test samples: {len(X_test)}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Time per sample: {(eval_time / len(X_test)) * 1000:.2f} ms")
    print("=" * 50)
    writer.add_scalar("benchmark/accuracy", test_accuracy, 0)
    writer.add_scalar("benchmark/time_seconds", eval_time, 0)
    writer.add_scalar(
        "benchmark/time_per_sample_ms", (eval_time / len(X_test)) * 1000, 0
    )
    hparams = {"model_path": str(model_path_obj)}
    metrics = {
        "hparam/accuracy": test_accuracy,
        "hparam/time_seconds": eval_time,
    }
    writer.add_hparams(hparams, metrics)
    writer.close()

    return {
        "accuracy": test_accuracy,
        "evaluation_time": eval_time,
        "num_samples": len(X_test),
        "log_dir": log_dir,
    }


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "benchmark", help="Benchmark trained SVM on CIFAR-10 test set"
    )
    parser.add_argument(
        "--model-path",
        default=SVM_CLASSIFIER_PATH,
        help="Path to the trained SVM model",
    )
    parser.set_defaults(entry=lambda args: run(model_path=args.model_path))
    return parser
