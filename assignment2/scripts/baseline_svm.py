from sklearn.metrics import classification_report
from sklearn.svm import SVC
from lib.data_utils import RESULTS_DIR
from lib.data_utils import get_train_test_split, save_results


def main():
    X_tr, X_te, y_tr, y_te = get_train_test_split()

    svm = SVC(C=0.01, kernel="linear", probability=True)
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_te)

    report_dict = classification_report(y_te, y_pred, output_dict=True)

    save_results(
        {
            "model": "SVC",
            "params": {"C": 0.01, "kernel": "linear", "probability": True},
            "report": report_dict,
        },
        RESULTS_DIR / "baseline.json",
    )


if __name__ == "__main__":
    main()
