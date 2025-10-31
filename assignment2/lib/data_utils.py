from pathlib import Path
from typing import Tuple
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = CACHE_DIR / "results"
DATA_DIR = CACHE_DIR / "dataset"
FIG_DIR = CACHE_DIR / "figures"


def load_full_dataset() -> tuple[pd.DataFrame, pd.Series]:
    def load_optdigits_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(path, header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    X_train, y_train = load_optdigits_csv(DATA_DIR / "optdigits.tra")
    X_test, y_test = load_optdigits_csv(DATA_DIR / "optdigits.tes")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    return X_full, y_full


def get_train_test_split(
    test_size: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_full, y_full = load_full_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        stratify=y_full,
    )
    return X_tr, X_te, y_tr, y_te


def save_results(data: dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote results to {filepath}")


def load_results(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)
