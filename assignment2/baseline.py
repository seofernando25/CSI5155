from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from data_utils import TRAIN_FILE, TEST_FILE, load_optdigits_csv

OUTPUT_JSON = Path(__file__).parent / "baseline.json"


def main():
    X_train_file, y_train_file = load_optdigits_csv(TRAIN_FILE)
    X_test_file, y_test_file = load_optdigits_csv(TEST_FILE)

    X_full = pd.concat([X_train_file, X_test_file])
    y_full = pd.concat([y_train_file, y_test_file])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.30, stratify=y_full
    )

    clf = LinearSVC(C=0.01, dual=False)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    report_dict = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_te, y_pred)

    payload = {
        "model": "LinearSVC",
        "params": {"C": 0.01, "kernel": "linear", "dual": False},
        "accuracy": accuracy,
        "report": report_dict,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote baseline metrics to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
