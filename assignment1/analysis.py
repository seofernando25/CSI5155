#!/usr/bin/env python3
"""Concise console summary of the marketing campaign dataset.

This script mirrors the key findings from `analysis.ipynb` without generating
plots. It focuses on the essentials needed downstream:

- Dataset shape and feature type counts
- Missing data and constant-feature checks
- Distribution skew for numeric features (proxy for normality)
- High-correlation feature pairs
- Cardinality of categoricals
- Class balance for binary flags (especially `Complain`)
- Quick summary of engineered aggregates
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from preprocessing import split_features

data_path = Path(__file__).resolve().parent / "marketing_campaign.csv"
max_corr_pairs = 10
skew_tolerance = 0.5


def format_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def describe_missing(df: pd.DataFrame) -> None:
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing == 0:
        print("No missing values detected.")
        return

    missing_pct = (missing_counts / len(df) * 100).round(2)
    nonzero = (
        pd.DataFrame({"count": missing_counts, "pct": missing_pct})
        .query("count > 0")
        .sort_values("count", ascending=False)
    )
    for col, row in nonzero.iterrows():
        print(f"{col}: {row['count']} missing ({row['pct']:.2f}% of rows)")
    print(f"Total missing entries: {total_missing}")


def describe_constants(df: pd.DataFrame) -> Iterable[str]:
    constants = df.columns[df.nunique(dropna=False) <= 1].tolist()
    if constants:
        print("Columns with constant values: " + ", ".join(constants))
    else:
        print("No constant-value columns detected.")
    return constants


def describe_numerics(df: pd.DataFrame, numeric_cols: Iterable[str]) -> Tuple[list[str], list[str]]:
    if not numeric_cols:
        print("No numeric features found.")
        return [], []

    skewness = df[numeric_cols].skew(numeric_only=True)
    approx_normal = skewness[skewness.abs() < skew_tolerance].sort_values()
    non_normal = skewness[skewness.abs() >= skew_tolerance].sort_values(key=np.abs, ascending=False)

    print(f"Numeric features: {len(numeric_cols)}")
    print(
        "Approximately symmetric distributions (|skew| < 0.5): "
        + (", ".join(approx_normal.index.tolist()) if not approx_normal.empty else "none")
    )
    print(
        "Most skewed features: "
        + ", ".join(f"{col} (skew={val:.2f})" for col, val in non_normal.head(5).items())
    )
    return approx_normal.index.tolist(), non_normal.index.tolist()


def describe_correlations(df: pd.DataFrame, numeric_cols: Iterable[str]) -> None:
    if not numeric_cols:
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    triu = np.triu_indices_from(corr, k=1)
    pairs = (
        pd.DataFrame(
            {
                "feature_1": corr.index.take(triu[0]),
                "feature_2": corr.columns.take(triu[1]),
                "abs_corr": np.abs(corr.values[triu]),
            }
        )
        .sort_values("abs_corr", ascending=False)
        .head(max_corr_pairs)
    )
    if pairs.empty:
        print("No correlated feature pairs identified.")
        return

    print(f"Top {len(pairs)} correlated numeric pairs (|ρ|):")
    for _, row in pairs.iterrows():
        print(f"{row['feature_1']} ↔ {row['feature_2']}: {row['abs_corr']:.3f}")


def describe_categoricals(df: pd.DataFrame, categorical_cols: Iterable[str]) -> None:
    if not categorical_cols:
        print("No categorical features found.")
        return

    card = df[categorical_cols].nunique(dropna=True).sort_values(ascending=False)
    for col, unique in card.items():
        print(f"{col}: {unique} unique values")


def describe_binaries(df: pd.DataFrame, binary_cols: Iterable[str]) -> None:
    if not binary_cols:
        print("No binary features detected.")
        return

    for col in binary_cols:
        counts = df[col].value_counts(dropna=False)
        total = counts.sum()
        ratios = (counts / total * 100).round(2)
        minority = ratios.min()
        values_summary = ", ".join(
            f"{val}: {counts[val]:>4} ({ratios[val]:>6.2f}%)" for val in counts.index
        )
        imbalance_flag = "⚠️" if minority < 20 else ("⚡" if minority < 30 else "✅")
        print(f"{col}: {values_summary} -> {imbalance_flag} minority share {minority:.2f}%")


def summarize_engineered_features(df: pd.DataFrame) -> None:
    engineered = [
        "Customer_Tenure",
        "TotalMnt",
        "TotalPurchases",
        "OnlinePurchaseShare",
        "CatalogPurchaseShare",
        "StorePurchaseShare",
        "TotalCampaignAccepts",
        "TotalKids",
    ]
    found = [col for col in engineered if col in df.columns]
    if not found:
        print("No engineered aggregate features detected.")
        return

    stats = df[found].describe(percentiles=[0.25, 0.5, 0.75]).T
    print(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2))


def main() -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not locate dataset at {data_path}")

    raw_df = pd.read_csv(data_path, sep="\t")
    X, y, numeric_cols, _categorical_cols = split_features(raw_df)
    df = X.copy()
    df["Complain"] = y

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = [col for col in df.columns if df[col].dropna().nunique() == 2]

    n_rows, n_cols = df.shape

    # Header
    format_section("Dataset Overview")
    print(f"Rows × Columns: {n_rows:,} × {n_cols}")
    print(f"Numeric features: {len(numeric_cols)}\tCategorical: {len(categorical_cols)}\tBinary: {len(binary_cols)}")

    # Missing values
    format_section("Missing Data")
    describe_missing(df)

    # Constant columns
    format_section("Constant Features")
    constants = describe_constants(df)

    # Distribution skew / normality proxy
    format_section("Numeric Distribution Summary")
    approx_normal, non_normal = describe_numerics(df, [c for c in numeric_cols if c not in constants])

    # Correlation insights
    format_section("Strong Correlations")
    describe_correlations(df, [c for c in numeric_cols if c not in constants])

    # Categorical cardinality
    format_section("Categorical Cardinality")
    describe_categoricals(df, categorical_cols)

    # Binary balance
    format_section("Binary Class Balance")
    describe_binaries(df, binary_cols)

    # Target imbalance focus
    if "Complain" in df.columns:
        counts = df["Complain"].value_counts()
        if len(counts) == 2 and counts.min() > 0:
            imbalance = counts.max() / counts.min()
            print(
                f"\nTarget 'Complain' imbalance: {counts.to_dict()} -> {imbalance:,.1f}:1 majority-to-minority ratio"
            )

    # Practical prep note
    format_section("Modeling Notes")
    print("- Impute numeric Income (24 missing) before modeling.")
    print("- Drop constants: Z_CostContact, Z_Revenue.")
    if approx_normal:
        print("- Only a handful of numeric features look roughly symmetric; use robust scalers.")
    else:
        print("- Numeric features are strongly skewed; prefer robust scaling or tree models.")
    high_imbalance_flags = [col for col in binary_cols if df[col].value_counts(normalize=True).min() < 0.2]
    if high_imbalance_flags:
        print("- Several binary features are highly imbalanced: " + ", ".join(high_imbalance_flags))
    print("- Consider deriving tenure from Dt_Customer (663 unique dates).")
    print("- Spending and purchase counts are tightly correlated; guard against multicollinearity in linear models.")

    format_section("Engineered Feature Summary")
    summarize_engineered_features(df)


if __name__ == "__main__":
    main()
