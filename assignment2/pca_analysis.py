import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data_utils import (
    FIG_DIR,
    TRAIN_FILE,
    TEST_FILE,
    load_optdigits_csv,
)

OUTPUT_FIG = FIG_DIR / "pca_scatter.png"
OUTPUT_GRID = FIG_DIR / "pca_grid.png"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_optdigits_csv(TRAIN_FILE)
    X_test, y_test = load_optdigits_csv(TEST_FILE)

    # Combine full dataset for PCA projection
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)

    expl_var = pca.explained_variance_ratio_
    print(
        f"Explained variance ratio by PC1 and PC2: {expl_var[0]:.4f}, {expl_var[1]:.4f}"
    )
    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    # Silhouette measures the quality of clustering: [-1, 1]
    sil_original = silhouette_score(X_full, y_full)
    print(f"Silhouette (original): {sil_original:.3f}")

    sil_pca = silhouette_score(X_pca, y_full)
    print(f"Silhouette (PCA 2D): {sil_pca:.3f}")

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for cluster_id in pd.unique(y_full):
        subset = X_pca[y_full == cluster_id]
        sns.kdeplot(
            x=subset[:, 0],
            y=subset[:, 1],
            fill=True,
            alpha=0.25,
            color=cmap(int(cluster_id)),
            thresh=0.1,
        )

    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y_full, cmap=cmap, s=10, alpha=0.5
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA representation")

    unique_labels = pd.unique(y_full)
    cbar = plt.colorbar(scatter, ticks=unique_labels)
    cbar.set_label("Classes")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=100)

    # 2x5 grid highlighting each class
    ax_main = plt.gca()
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    unique_sorted = sorted(unique_labels)

    faint_kde_alpha = 0.02
    faint_pts_alpha = 0.05
    normal_kde_alpha = 0.25
    normal_pts_alpha = 0.8

    fig_grid, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for idx, lbl in enumerate(unique_sorted):
        row = idx // 5
        col = idx % 5
        ax_cell = axes[row][col]
        ax_cell.set_xlim(xlim)
        ax_cell.set_ylim(ylim)
        for other in unique_sorted:
            other_int = int(other)
            subset = X_pca[y_full == other_int]
            is_current = other_int == int(lbl)
            alpha_kde = normal_kde_alpha if is_current else faint_kde_alpha
            alpha_pts = normal_pts_alpha if is_current else faint_pts_alpha
            colr = "black" if is_current else plt.get_cmap("tab10")(other_int)
            sns.kdeplot(
                x=subset[:, 0],
                y=subset[:, 1],
                fill=True,
                alpha=alpha_kde,
                color=colr,
                thresh=0.1,
                ax=ax_cell,
            )
            ax_cell.scatter(
                subset[:, 0], subset[:, 1], s=10, alpha=alpha_pts, color=colr
            )
        ax_cell.set_title(f"Class {int(lbl)}")

    for ax_cell in axes[-1]:
        ax_cell.set_xlabel("PC1")
    for ax_cell in axes[:, 0]:
        ax_cell.set_ylabel("PC2")
    fig_grid.tight_layout()
    fig_grid.savefig(OUTPUT_GRID, dpi=120)
    plt.close(fig_grid)


if __name__ == "__main__":
    main()
