# Assignment 2 – Semi-Supervised Learning (CSI5155 - Fall 2025)

**University of Ottawa**
**School of Electrical Engineering and Computer Science**

**Due date:** 31 October, 2025 (11:59pm ET)
**Total marks:** 100

## Instructions:
1. This is an **individual assignment**. Submit your assignment (source code, output, and written report) using uOttawa's BrightSpace before the due date.
2. Use **Scikit-Learn** to complete the assignment.
3. In addition to the source code and written report, all students are expected to demonstrate and explain their projects during a time slot the teaching assistant will schedule after submission. This demonstration will be used in conjunction with the submitted assignment for assignment evaluation.
4. Please refer to the AI Policy and Late Policy on the course syllabus.

## Description

This assignment considers the **Optical Recognition of Handwritten Digits dataset**, available here:
[https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

You may download the dataset directly, but it can also be imported directly as follows:

```python
from ucimlrepo import fetch_ucirepo
handwritten_digits = fetch_ucirepo(id=80)
X = handwritten_digits.data.features
Y = handwritten_digits.data.targets
```

The data set was created to train machine classifiers to recognize hand-written digits. The images were originally $32 \times 32$ pixels, but the dataset creators preprocessed them to reduce the size to $8 \times 8$ pixels. These square images can then be treated as a single feature vector of length 64. The features are integer-valued.

The entire dataset (5620 samples) is labelled. However, in this assignment we will remove the labels from subsets of the data to analyze the effectiveness of semi-supervised learning on this task.

---

**Examples of the downsampled ($8 \times 8$) images**

This section displays a grid of 10 small, grayscale images, each representing a handwritten digit from 0 to 9. Each image is labeled with its corresponding class (Label: [0] through Label: [9]). The images are $8 \times 8$ pixels, showing simple, blocky representations of the numbers.

---

## 1. Unsupervised learning: dimensionality reduction with PCA [10 marks]

Use PCA to create a two-dimensional representation of the data. Project the entire dataset onto the first two principal components and generate a scatter plot. The colour of each data point should represent its class ($0, 1, 2, \dots, 9$).

Visually, does it appear that the **Smoothness Assumption** is satisfied? That is, do points that are near to each other in the (transformed) input space also tend to belong to the same class?

## 2. Supervised learning baseline [10 marks]

**Training:** Divide the data into $70\%$ training data and $30\%$ test data (stratifying the classes). Train an **SVM classifier** on the entire labelled training set to establish **baseline performance**. You may skip hyperparameter optimization and simply use $C=0.01$ and a linear kernel.

**Evaluation:** Output the scikit-learn **"classification report"** for the classifier on the test set. This should include measures of precision, recall, F1 score, and accuracy per class, as well as averaged over the classes.

## 3. Semi-supervised learning [50 marks]

Now implement two semi-supervised approaches that we discussed in class, namely:
1. **`SelfTrainingClassifier()`** ([https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html))
2. **`LabelPropagation()`** ([https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html))

For `SelfTrainingClassifier` we will use **SVM** as the base model. You can again use $C = 0.01$ and `kernel='linear'`. You must also specify `probability=True` for the SVM classifier to output the confidence estimates needed by the `SelfTrainingClassifier`.

For the `SelfTrainingClassifier`, we also need to specify the criterion that we will use to add pseudo-labelled data to the training set (either based on a confidence threshold or a ranking procedure). Since SVM does not produce reliable confidence estimates, we will use the ranking method (`criterion = 'k_best'`). Also increase the maximum number of iterations to 100.

For `LabelPropagation`, we will use the nearest neighbour method (`kernel = 'knn'`) with the default number of neighbours, $k=7$.

***Note 1:*** *Normally, we would want to do hyperparameter tuning in the semi-supervised case as well. However, we will be working with such small quantities of labelled training data that it will be difficult to carve out a meaningful validation set for hyperparameter tuning, so we will skip that step in this assignment.*

### Training:

We want to investigate how performance changes depending on the amount of labelled data that we have available.

Modify the training set so that only a small proportion $p$ of the training data is labelled and the rest is unlabelled. Consider the following proportions:
$$p = 0.2\%, 0.4\%, 0.6\%, 0.8\%, 1\%, 1.5\%, 2\%, 2.5\%, 3\%, 4\%, 5\%, \text{ and } 10\% \text{ of the labels (12 datapoints altogether).}$$

For each $p$:
* Train the **SVM classifier** on only the labelled subset (supervised learning).
* Train the **`SelfTrainingClassifier`** on the semi-supervised training set.
* Train the **`LabelPropagation`** algorithm on the semi-supervised training set.

***Note 2:*** *As we learned in class, the `LabelPropagation` algorithm is **transductive**: it estimates labels for the unlabelled portion of the training set. However, scikit-learn also provides a `predict()` method for this classifier to perform inductive learning with the trained model on new data (i.e., the test set).*

### Evaluation:

Test the models on the held-out test set. For all three models, at all 12 label proportions $p$, determine the **accuracy, macro-F1, macro-recall, and macro-precision**.

### Visualization:
(1) For all three models, output a plot of **macro-F1 ($y$-axis) versus label proportion ($x$-axis)**.

(2) Some classes are harder to predict than others. For example, the digit "6" is usually consistently identified, even with small amounts of training data. On the other hand, the digit "8" seems to be much harder. (Does this align with what you observed in the PCA plot?) Output two separate plots that show the **(per-class) F1 for class 6 only ($y$-axis)**, and the **per-class F1 for class 8 only ($y$-axis)**, versus training label proportion ($x$-axis), for all three classifiers.

---

## 3. Reporting [30 marks]

Submit a **400 to 500 words** written summary, discussing the results you obtained and the lessons you learned when analysing this data. For example:
* How did the semi-supervised methods compare to the fully-supervised baseline?
* Which semi-supervised algorithm performed better? Does your answer change depending on whether you have 1% labelled data or 10% labelled data? Does your answer change depending on which class you look at?
* Do you have any ideas how you might improve performance in the future?