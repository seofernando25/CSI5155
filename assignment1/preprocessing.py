#!/usr/bin/env python3
"""
Feature engineering and data preprocessing pipeline.

This module handles all data preprocessing including:
- Loading the dataset
- Feature engineering (tenure, aggregates, ratios)
- Train/test splitting
- Column type identification
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
DATA_PATH = 'marketing_campaign.csv'
TARGET_COL = 'Complain'
CONSTANT_COLS = {'Z_CostContact', 'Z_Revenue'}
IDENTIFIER_COLS = {'ID'}


def load_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the marketing campaign dataset."""
    return pd.read_csv(path, sep='\t')


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from the raw dataset.
    
    Features created:
    - Customer_Tenure: days since customer registration
    - TotalMnt: total spending across all product categories
    - TotalPurchases: total number of purchases
    - Purchase share ratios (online, catalog, store)
    - TotalCampaignAccepts: total campaign acceptances
    - TotalKids: total children at home
    """
    X = X.copy()
    
    # Derive tenure from Dt_Customer
    if 'Dt_Customer' in X.columns:
        try:
            dt_series = pd.to_datetime(X['Dt_Customer'], dayfirst=False, errors='coerce')
            max_date = dt_series.max()
            X['Customer_Tenure'] = (max_date - dt_series).dt.days
        except Exception:
            X['Customer_Tenure'] = 0
        X = X.drop(columns=['Dt_Customer'])
    
    # Total spending
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    if all(col in X.columns for col in spend_cols):
        X['TotalMnt'] = X[spend_cols].sum(axis=1)
    
    # Total purchases and purchase shares
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in X.columns for col in purchase_cols):
        total_purchases = X[purchase_cols].sum(axis=1)
        X['TotalPurchases'] = total_purchases
        
        with np.errstate(divide='ignore', invalid='ignore'):
            total_nonzero = total_purchases.replace(0, np.nan)
            X['OnlinePurchaseShare'] = (X['NumWebPurchases'] / total_nonzero).fillna(0.0)
            X['CatalogPurchaseShare'] = (X['NumCatalogPurchases'] / total_nonzero).fillna(0.0)
            X['StorePurchaseShare'] = (X['NumStorePurchases'] / total_nonzero).fillna(0.0)
    
    # Total campaign accepts
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                     'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    present_campaigns = [c for c in campaign_cols if c in X.columns]
    if present_campaigns:
        X['TotalCampaignAccepts'] = X[present_campaigns].sum(axis=1)
    
    # Total kids
    if 'Kidhome' in X.columns and 'Teenhome' in X.columns:
        X['TotalKids'] = X['Kidhome'] + X['Teenhome']
    
    return X


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Split dataframe into features (X) and target (y), with feature engineering.
    
    Returns:
        X: Feature dataframe
        y: Target series
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f'TARGET_COL {TARGET_COL} not in dataframe')
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    
    # Engineer features
    X = engineer_features(X)
    
    # Drop constant columns
    drop_cols = [c for c in X.columns 
                 if c in CONSTANT_COLS or c in IDENTIFIER_COLS 
                 or X[c].nunique(dropna=False) <= 1]
    
    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], 
                       categorical_cols: List[str], 
                       scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a preprocessing pipeline.
    
    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        scale_numeric: Whether to apply StandardScaler to numeric features
    
    Returns:
        ColumnTransformer with preprocessing steps
    """
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

