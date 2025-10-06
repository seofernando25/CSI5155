import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import load_dataframe, split_features
from .theme_config import FIGURE_CONFIG, FONT_CONFIG, FIGURE_SIZES, HISTOGRAM_CONFIG


def analyze_engineered_features():
    df = load_dataframe()
    X, y, numeric_cols, categorical_cols = split_features(df)

    original_cols = set(df.columns) - {
        "Complain",
        "Dt_Customer",
        "ID",
        "Z_CostContact",
        "Z_Revenue",
    }
    engineered_cols = [col for col in X.columns if col not in original_cols]

    return X, y, engineered_cols, numeric_cols


def generate_engineered_features_histograms(X, engineered_cols, save_dir="figures"):
    """Generate histograms for engineered features."""
    os.makedirs(save_dir, exist_ok=True)


    # Filter out binary features
    continuous_engineered = []
    for col in engineered_cols:
        if col in X.columns:
            unique_vals = sorted(X[col].unique())
            unique_count = len(unique_vals)
            std_val = X[col].std()

            # Skip if binary (0/1) or constant
            is_binary = unique_count == 2 and set(int(x) for x in unique_vals) == {0, 1}

            if unique_count > 2 and not is_binary and std_val > 0:
                continuous_engineered.append(col)


    # Create figure
    n_cols = 3
    n_rows = (len(continuous_engineered) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGURE_SIZES["medium"])
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, col in enumerate(continuous_engineered):
        if i < len(axes):
            ax = axes[i]

            data = X[col].dropna()
            ax.hist(data, **HISTOGRAM_CONFIG)
            ax.set_title(
                f"{col}\n(Skew: {data.skew():.3f})", fontsize=FONT_CONFIG["label"]
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            mean_val = data.mean()
            ax.axvline(
                mean_val, linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.2f}"
            )
            ax.legend(fontsize=FONT_CONFIG["legend"])

    for i in range(len(continuous_engineered), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/engineered_features_histograms.png", **FIGURE_CONFIG)
    plt.close()

def generate_engineered_features_correlations(
    X, engineered_cols, y, save_dir="figures"
):
    """Generate correlation matrix for engineered features."""
    os.makedirs(save_dir, exist_ok=True)


    # Get engineered features that exist in the dataset
    existing_engineered = [col for col in engineered_cols if col in X.columns]

    # Create correlation matrix with target
    corr_data = X[existing_engineered].copy()
    corr_data["Complain"] = y

    corr_matrix = corr_data.corr()

    # Create correlation matrix heatmap
    plt.figure(figsize=FIGURE_SIZES["small"])

    # Create mask to hide upper triangle but keep diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    np.fill_diagonal(mask, False)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
    )

    plt.title(
        "Engineered Features Correlation Matrix",
        fontsize=FONT_CONFIG["subtitle"],
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/engineered_features_correlation_matrix.png", **FIGURE_CONFIG
    )
    plt.close()

def main():
    print("Engineered features analysis")

    X, y, engineered_cols, _numeric_cols = analyze_engineered_features()

    generate_engineered_features_histograms(X, engineered_cols)
    generate_engineered_features_correlations(X, engineered_cols, y)


if __name__ == "__main__":
    main()
