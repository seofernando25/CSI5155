from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import ClassifierMixin
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sampling import SamplingStrategy, get_sampler

DATA_PATH = "marketing_campaign.csv"
TARGET_COL = "Complain"
DROPPED_COLS = {
    "Z_CostContact",  # Always 3 in dataset
    "Z_Revenue",  # ALWAYS 11 in dataset
    "ID",  # Not useful
}

def load_dataframe(path: str = DATA_PATH):
    return pd.read_csv(path, sep="\t")


def engineer_features(X: pd.DataFrame):
    X = X.copy()

    dt_series = pd.to_datetime(X["Dt_Customer"], dayfirst=True)
    max_date = dt_series.max()
    X["Customer_Tenure"] = (max_date - dt_series).dt.days
    X = X.drop(columns=["Dt_Customer"])

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    X["TotalMnt"] = X[spend_cols].sum(axis=1)

    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    total_purchases = X[purchase_cols].sum(axis=1)
    X["TotalPurchases"] = total_purchases

    # replace 0 to avoid division by 0
    # still should work because 0/1 = 0
    X["OnlinePurchaseShare"] = X["NumWebPurchases"] / total_purchases.replace(0, 1)
    X["CatalogPurchaseShare"] = X["NumCatalogPurchases"] / total_purchases.replace(0, 1)
    X["StorePurchaseShare"] = X["NumStorePurchases"] / total_purchases.replace(0, 1)

    campaign_cols = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response",
    ]
    X["TotalCampaignAccepts"] = X[campaign_cols].sum(axis=1)
    
    X["TotalKids"] = X["Kidhome"] + X["Teenhome"]

    return X


def split_features(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL {TARGET_COL} not in dataframe")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    X = engineer_features(X)
    X = X.drop(columns=DROPPED_COLS, errors="ignore")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor

def build_pipeline(
    preprocessor: ColumnTransformer,
    classifier: ClassifierMixin,
    sampling: SamplingStrategy = "none",
):
    sampler = get_sampler(sampling)
    steps = [("pre", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", classifier))
    return Pipeline(steps=steps)
