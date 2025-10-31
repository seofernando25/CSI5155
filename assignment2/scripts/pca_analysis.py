import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from lib.data_utils import FIG_DIR, load_full_dataset


def main():
    PCA_ANALYSIS_FIG_DIR = FIG_DIR / "pca_analysis"
    PCA_ANALYSIS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_full, y_full = load_full_dataset()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)

    expl_var = pca.explained_variance_ratio_
    print(
        f"Explained variance ratio by PC1 and PC2: {expl_var[0]:.4f}, {expl_var[1]:.4f}"
    )
    sil_original = silhouette_score(X_full, y_full)
    print(f"Silhouette (original): {sil_original:.3f}")

    sil_pca = silhouette_score(X_pca, y_full)
    print(f"Silhouette (PCA 2D): {sil_pca:.3f}")

    plt.figure()
    cmap = plt.get_cmap("tab10")

    for cluster_id in pd.unique(y_full):
        subset = X_pca[y_full == cluster_id]
        sns.kdeplot(
            x=subset[:, 0],
            y=subset[:, 1],
            fill=True,
            alpha=0.25,
            color=cmap(cluster_id),
            thresh=0.1,
        )

    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y_full, cmap=cmap, s=10, alpha=0.5
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA representation")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Classes")
    plt.tight_layout()
    plt.savefig(PCA_ANALYSIS_FIG_DIR / "pca_scatter.png")

    ax_main = plt.gca()
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    unique_sorted = pd.unique(y_full)
    unique_sorted.sort()

    faint_pts_alpha = 0.05
    normal_pts_alpha = 0.8

    fig_grid, axes = plt.subplots(
        2, 5, figsize=(5 * 3, 2 * 3), sharex=True, sharey=True
    )

    for idx, lbl in enumerate(unique_sorted):
        row = idx // 5
        col = idx % 5
        ax_cell = axes[row][col]
        ax_cell.set_xlim(xlim)
        ax_cell.set_ylim(ylim)
        for other in unique_sorted:
            subset = X_pca[y_full == other]
            is_current = other == lbl
            alpha_pts = normal_pts_alpha if is_current else faint_pts_alpha
            colr = "black" if is_current else plt.get_cmap("tab10")(other)
            ax_cell.scatter(
                subset[:, 0], subset[:, 1], s=10, alpha=alpha_pts, color=colr
            )
        ax_cell.set_title(f"Class {int(lbl)}")

    for ax_cell in axes[-1]:
        ax_cell.set_xlabel("PC1")
    for ax_cell in axes[:, 0]:
        ax_cell.set_ylabel("PC2")
    fig_grid.tight_layout()
    fig_grid.savefig(PCA_ANALYSIS_FIG_DIR / "pca_grid.png")
    plt.close(fig_grid)


if __name__ == "__main__":
    main()
