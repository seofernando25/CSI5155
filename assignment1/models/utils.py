#!/usr/bin/env python3
import os
import json
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'marketing_campaign.csv')
TARGET_COL = 'Complain'
CONSTANT_COLS = {'Z_CostContact', 'Z_Revenue'}
IDENTIFIER_COLS = {'ID'}
SAMPLING_STRATEGIES = ('none', 'under', 'smote', 'over')


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep='\t')
    return df


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if TARGET_COL not in df.columns:
        raise KeyError(f'TARGET_COL {TARGET_COL} not in dataframe')
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # drop constant columns if present
    drop_cols = [c for c in X.columns if c in CONSTANT_COLS or X[c].nunique(dropna=False) <= 1]
    # drop identifiers and derive tenure from Dt_Customer
    tenure_col = 'Customer_Tenure'
    if 'Dt_Customer' in X.columns:
        try:
            dt_series = pd.to_datetime(X['Dt_Customer'], dayfirst=False, errors='coerce')
            max_date = dt_series.max()
            X[tenure_col] = (max_date - dt_series).dt.days
        except Exception:
            # fall back to zero if parsing fails
            X[tenure_col] = 0
        drop_cols.append('Dt_Customer')

    spend_cols = [
        'MntWines',
        'MntFruits',
        'MntMeatProducts',
        'MntFishProducts',
        'MntSweetProducts',
        'MntGoldProds',
    ]
    if all(col in X.columns for col in spend_cols):
        X['TotalMnt'] = X[spend_cols].sum(axis=1)

    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in X.columns for col in purchase_cols):
        total_purchases = X[purchase_cols].sum(axis=1)
        X['TotalPurchases'] = total_purchases
        with np.errstate(divide='ignore', invalid='ignore'):
            total_nonzero = total_purchases.replace(0, np.nan)
            X['OnlinePurchaseShare'] = (X['NumWebPurchases'] / total_nonzero).fillna(0.0)
            X['CatalogPurchaseShare'] = (X['NumCatalogPurchases'] / total_nonzero).fillna(0.0)
            X['StorePurchaseShare'] = (X['NumStorePurchases'] / total_nonzero).fillna(0.0)

    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    present_campaigns = [c for c in campaign_cols if c in X.columns]
    if present_campaigns:
        X['TotalCampaignAccepts'] = X[present_campaigns].sum(axis=1)

    if 'Kidhome' in X.columns and 'Teenhome' in X.columns:
        X['TotalKids'] = X['Kidhome'] + X['Teenhome']

    drop_cols.extend([c for c in IDENTIFIER_COLS if c in X.columns])
    drop_cols = list(dict.fromkeys(drop_cols))

    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))

    categorical_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(numeric_steps), numeric_cols),
            ('cat', Pipeline(categorical_steps), categorical_cols),
        ],
        remainder='drop',
        sparse_threshold=0.3,
    )
    return preprocessor


def get_sampler(sampling: str):
    if sampling is None or sampling == 'none':
        return None
    if sampling == 'smote':
        return SMOTE(random_state=42, k_neighbors=1)
    if sampling == 'over':
        return RandomOverSampler(random_state=42)
    if sampling == 'under':
        return RandomUnderSampler(random_state=42)
    raise ValueError(f"Unknown sampling: {sampling}")


def build_pipeline(preprocessor: ColumnTransformer, classifier, sampling: str):
    sampler = get_sampler(sampling)
    steps = [('pre', preprocessor)]
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('clf', classifier))
    return ImbPipeline(steps)


def evaluate_and_report(model_name: str, estimator: Pipeline, X: pd.DataFrame, y: pd.Series, out_dir: str, cv_splits: int = 5) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # Out-of-fold probabilities and predictions
    probas = cross_val_predict(estimator, X, y, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42), method='predict_proba', n_jobs=-1)[:, 1]
    y_pred = (probas >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, probas)
    auc = roc_auc_score(y, probas)

    # Save metrics
    metrics = {
        'model': model_name,
        'precision': float(prec),
        'recall': float(rec),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()

    # Print brief report
    print(f'[{model_name}] precision={prec:.4f} recall={rec:.4f} auc={auc:.4f}')
    print('Confusion Matrix:\n', cm)

    return metrics
