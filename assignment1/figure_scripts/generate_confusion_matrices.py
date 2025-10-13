import numpy as np
from .theme_config import FONT_CONFIG, MODEL_LABELS, SAMPLINGS
from .common_utils import (
    load_tuning_results,
    find_metrics_for_model,
    setup_subplot_grid,
    add_suptitle,
    save_figure,
)


def generate_confusion_matrix_grid(
    output_dir="figures", tuning_file="comprehensive_evaluation_results.json", cv_type="cv2"
):
    tuning_results = load_tuning_results(tuning_file)
    if tuning_results is None:
        return

    for sampling_key, sampling_label in SAMPLINGS.items():
        fig, axes = setup_subplot_grid(2, 3, "large")

        for idx, (model_key, model_label) in enumerate(MODEL_LABELS.items()):
            ax = axes[idx]

            metrics = find_metrics_for_model(tuning_results, model_key, sampling_key, cv_type)

            if metrics is not None:
                cm = np.array(metrics["confusion_matrix"])

                ax.imshow(cm, cmap='Blues')
                ax.set_title(
                    f"{model_label}\n(P={metrics['precision']:.3f}, R={metrics['recall']:.3f})",
                    fontsize=FONT_CONFIG["label"],
                    fontweight="bold",
                )

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j,
                            i,
                            int(cm[i, j]),
                            ha="center",
                            va="center",
                            color="black",
                            bbox=dict(facecolor="white", alpha=0.8),
                            fontsize=FONT_CONFIG["annotation"]
                        )

                ax.set_xlabel("Predicted", fontsize=FONT_CONFIG["tick"])
                ax.set_ylabel("Actual", fontsize=FONT_CONFIG["tick"])
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["No Complaint", "Complaint"])
                ax.set_yticklabels(["No Complaint", "Complaint"])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Data Not Available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(
                    model_label, fontsize=FONT_CONFIG["label"], fontweight="bold"
                )

        add_suptitle(fig, f"Confusion Matrices - {sampling_label} ({cv_type.upper()})")
        save_figure(f"confusion_matrices_{sampling_key}.png", output_dir)


def main():
    """Main entry point for confusion matrices generation."""
    generate_confusion_matrix_grid()


if __name__ == "__main__":
    main()
