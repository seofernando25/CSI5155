import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import load_dataframe, TARGET_COL
from .theme_config import FIGURE_CONFIG, FONT_CONFIG, FIGURE_SIZES, HISTOGRAM_CONFIG


def generate_histograms(df, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    continuous_cols = []
    excluded_cols = []

    for col in numeric_cols:
        if col == "ID":
            excluded_cols.append(col)
            continue

        unique_vals = sorted(df[col].unique())
        unique_count = len(unique_vals)
        std_val = df[col].std()

        # ignore const and binary from histograms
        if unique_count <= 2:
            excluded_cols.append(col) 
        elif std_val == 0:
            excluded_cols.append(col) 
        else:
            continuous_cols.append(col)

    # just to auto wrap plots
    n_cols = 4
    n_rows = (len(continuous_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGURE_SIZES["wide"], squeeze=False)
    axes = axes.ravel()

    for i, col in enumerate(continuous_cols):
        if i < len(axes):
            ax = axes[i]

            data = df[col].dropna()
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

    

    plt.tight_layout()
    plt.savefig(f"{save_dir}/continuous_features_histograms.png", **FIGURE_CONFIG)
    plt.close()


def generate_target_distribution(df, save_dir="figures"):
    """Generate target variable distribution plot."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["medium"])

    # Count plot
    counts = df[TARGET_COL].value_counts()
    ax1.bar(counts.index, counts.values)
    ax1.set_title("Target Variable Distribution (Counts)")
    ax1.set_xlabel(TARGET_COL)
    ax1.set_ylabel("Count")

    # Add count labels on bars
    for i, (idx, val) in enumerate(counts.items()):
        ax1.text(
            idx,
            val + max(counts.values) * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Percentage plot
    percentages = df[TARGET_COL].value_counts(normalize=True) * 100
    ax2.bar(percentages.index, percentages.values)
    ax2.set_title("Target Variable Distribution (Percentages)")
    ax2.set_xlabel(TARGET_COL)
    ax2.set_ylabel("Percentage (%)")

    for i, (idx, val) in enumerate(percentages.items()):
        ax2.text(
            idx,
            val + max(percentages.values) * 0.01,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/target_distribution.png", **FIGURE_CONFIG)
    plt.close()


def generate_binary_features_plots(df, save_dir="figures"):
    """Generate pie charts for binary features."""
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Find binary features (0/1 values)
    binary_cols = []
    for col in numeric_cols:
        unique_vals = sorted(df[col].unique())
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            binary_cols.append(col)

    n_cols = 4
    n_rows = (len(binary_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGURE_SIZES["medium"])
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, col in enumerate(binary_cols):
        if i < len(axes):
            ax = axes[i]

            # Get counts
            counts = df[col].value_counts()
            labels = [f"{val} ({count})" for val, count in counts.items()]

            # Create pie chart
            ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{col}", fontsize=12, fontweight="bold")

    # Hide empty subplots
    for i in range(len(binary_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/binary_features_pie_charts.png", **FIGURE_CONFIG)
    plt.close()


def generate_missing_data_analysis(df, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    total_rows = len(df)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZES["small"])

    income_missing = missing_data.iloc[0]
    income_existing = total_rows - income_missing

    sizes = [income_existing, income_missing]
    labels = [
        f"Existing Data\n({income_existing:,} values)",
        f"Missing Data\n({income_missing} values)",
    ]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.2f%%",
        startangle=90,
        textprops={"fontsize": FONT_CONFIG["label"], "fontweight": "bold"},
    )

    # Customize the percentage text
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(FONT_CONFIG["subtitle"])
        autotext.set_fontweight("bold")

    ax.set_title("Income", fontsize=FONT_CONFIG["title"], fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/missing_data_analysis.png", **FIGURE_CONFIG)
    plt.close()

def generate_correlation_matrix(df, save_dir="figures"):
    """Generate correlation matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    exclude_cols = ["ID"]
    constant_cols = []
    for col in numeric_cols:
        if df[col].std() == 0:
            constant_cols.append(col)

    exclude_cols.extend(constant_cols)
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(20, 16))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
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
        annot_kws={"size": 6},
        xticklabels=True,
        yticklabels=True,
    )

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.title(
        "Feature Correlation Matrix",
        fontsize=FONT_CONFIG["title"],
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_matrix.png", **FIGURE_CONFIG)
    plt.close()

    if TARGET_COL in corr_matrix.columns:
        target_corr = (
            corr_matrix[TARGET_COL]
            .drop(TARGET_COL)
            .sort_values(key=abs, ascending=False)
        )

        plt.figure(figsize=(12, 14))
        bars = plt.barh(range(len(target_corr)), target_corr.values)
        plt.yticks(range(len(target_corr)), target_corr.index, fontsize=9)
        plt.xlabel(f"Correlation with {TARGET_COL}", fontsize=FONT_CONFIG["tick"])
        plt.title(
            f"Feature Correlations with Target Variable ({TARGET_COL})",
            fontsize=FONT_CONFIG["subtitle"],
            fontweight="bold",
        )
        plt.grid(axis="x")

        # Add correlation values on bars
        for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
            plt.text(
                val + (0.01 if val >= 0 else -0.01),
                i,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/target_correlations.png", **FIGURE_CONFIG)
        plt.close()



def main():
    print("Dataset analysis")

    df = load_dataframe()

    generate_histograms(df)
    generate_binary_features_plots(df)
    generate_correlation_matrix(df)
    generate_target_distribution(df)
    generate_missing_data_analysis(df)

    print("Dataset analysis complete!")


if __name__ == "__main__":
    main()
