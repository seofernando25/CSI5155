# Report

## PCA Analysis

PCA was used to project the 64-dimensional data onto a 2D space. The visualization shows that nearby points tend to share labels, confirming **the Smoothness Assumption is satisfied**.

![PCA scatter plot](.cache/figures/pca_analysis/pca_scatter.png)

![PCA grid](.cache/figures/pca_analysis/pca_grid.png)


## Supervised Learning Baseline

A Support Vector Machine (SVM) classifier with a linear kernel and C=0.01 was trained on the entire 70% labeled training dataset to establish a baseline for performance. This represents the performance of a standard supervised learning approach with a sufficient amount of labeled data. The classification report on the 30% test set is as follows:

```
class         precision     recall   f1-score    support
--------------------------------------------------------
0                  0.99       1.00       0.99        166
1                  0.94       0.98       0.96        171
2                  1.00       0.99       0.99        167
3                  0.97       0.98       0.97        172
4                  0.98       0.98       0.98        170
5                  0.98       0.98       0.98        168
6                  0.98       0.98       0.98        167
7                  0.98       0.99       0.99        170
8                  0.97       0.94       0.96        166
9                  0.98       0.93       0.95        169
--------------------------------------------------------
accuracy                                 0.98       1686
macro avg          0.98       0.98       0.98       1686
weighted avg       0.98       0.98       0.98       1686
```

I have also calculated the silhouette score, which quantifies this observation, showing the original 64D space is more structurally coherent (0.33) than the 2D projection (0.27). But even so, the underlying cluster structure is still mostly preserved.

## Semi-Supervised Learning Results

To test the effectiveness of SSL, I compared the three models across 12 label proportions (p=0.2% to 10%) on a 70/30 stratified train/test split. 

The models were:
- A supervised SVM baseline (linear kernel, C=0.01).
- A SelfTrainingClassifier using the same SVM as its base.
- A LabelPropagation model (kernel='knn', n_neighbors=7). 

Performance was measured by accuracy, macro-F1, macro-precision, and macro-recall scores on the test set.

### Overall Performance

The following plots show the performance of the two semi-supervised methods against the fully supervised baseline.

![Accuracy vs label proportion](.cache/figures/semi_supervised/accuracy.png)
![Macro-F1 vs label proportion](.cache/figures/semi_supervised/macro_f1.png)
![Macro-Precision vs label proportion](.cache/figures/semi_supervised/macro_precision.png)
![Macro-Recall vs label proportion](.cache/figures/semi_supervised/macro_recall.png)

### Per-Class F1-Score for Classes "6" and "8"

![F1 class 6 vs p](.cache/figures/semi_supervised/f1_class_6.png)
![F1 class 8 vs p](.cache/figures/semi_supervised/f1_class_8.png)

## Conclusion & Discussion

These results demonstrate the effectiveness of semi-supervised learning when labeled data is limited. We can now address the main questions as follows:

### Comparison to Supervised Baseline

As expected, the fully supervised SVM baseline, trained on all labeled data, achieves a high macro F1-score of 0.98, representing the upper bound for performance in this setup.

The semi-supervised methods, `LabelPropagation` and `SelfTrainingClassifier`, start with much lower performance at very small label proportions but improve as more labeled data is introduced. `LabelPropagation` shows a clear advantage, consistently outperforming `SelfTrainingClassifier` across all metrics and proportions. At p=10%, `LabelPropagation`'s performance becomes very close to the fully supervised baseline, demonstrating its ability to effectively leverage unlabeled data. For instance, its macro F1-score reaches ~0.98, nearly matching the baseline. `SelfTrainingClassifier` also improves significantly but does not reach the same level of performance, capping at a macro F1-score of ~0.95.

**Semi-Supervised Algorithm Performance:**
Overall, **`LabelPropagation` was the superior semi-supervised algorithm**. Its graph-based nature aligns well with the data's manifold structure, making it robust and effective even with few labels. The `SelfTrainingClassifier` struggled at very low proportions of labeled data. This was particularly evident for the challenging digit "8." At p=0.4% `SelfTrainingClassifier` had an F1-score of 0.36 for this class, while `LabelPropagation` failed to classify it correctly at all (F1-score of 0.00). This suggests that for some classes, `SelfTrainingClassifier` can learn better from very few examples, while `LabelPropagation` may need more labeled data to get started. However, with more labeled data (`p=10%`), `LabelPropagation` becomes much better for class 8 as well, reaching an F1-score of 0.96.

**Ideas for Improvement:**
Future performance could be enhanced by:
*   Careful **hyperparameter tuning** (e.g., SVM's `C` or `n_neighbors` for `LabelPropagation`).
*   **Calibrating the SVM's probability outputs** to provide more reliable confidence scores for the `SelfTrainingClassifier`.
*   Implementing better data strategies, such as **class-balanced initial sampling** or **data augmentation**.