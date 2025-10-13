from pathlib import Path
from typing import Tuple

import pandas as pd

# Common paths
DATA_DIR = Path(__file__).parent / "dataset"
TRAIN_FILE = DATA_DIR / "optdigits.tra"
TEST_FILE = DATA_DIR / "optdigits.tes"
FIG_DIR = Path(__file__).parent / "figures"


def load_optdigits_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a UCI optdigits CSV (no header): returns (X, y)."""
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def load_full_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load train and test CSVs and concatenate into a full dataset."""
    X_train, y_train = load_optdigits_csv(TRAIN_FILE)
    X_test, y_test = load_optdigits_csv(TEST_FILE)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    return X_full, y_full
