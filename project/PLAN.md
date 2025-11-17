# Strategic Plan for CSI 5155 Project: From Classical CV to Double Descent


## Thesis

The new thesis is: *This project investigates the spectrum of model generalization, from the classical, feature-engineered regime (SVM), through the "classical" deep learning regime (an optimized CNN), and into the "modern" over-parameterized regime. In this final stage, the double descent phenomenon will be experimentally induced and analyzed to explore the theoretical paradoxes of deep learning generalization.*


**4.1. The Central Figure: The Double Descent Curve**

- Primary result: a line plot with:
    - X-axis: Model Width (k) or Total Parameters (log scale).
    - Y-axis: Test Set Error (%).
- This should show the full double descent curve.
- Identify three key points for further analysis:
    - CNN(k=small): Best in "first descent".
    - CNN(k=critical): At error peak (interpolation threshold).
    - CNN(k=overparam): Best in "second descent".

**4.2. Analysis 1: Statistical Significance (Rubric Criterion 1)**

- Use McNemar's Test (not a t-test) for comparing classifiers on the same test set.
- **Action Plan:**
    - Run McNemar's Test on the test set for:
        - Pair A: SVM Baseline vs. CNN(k=overparam).
        - Pair B: CNN(k=critical) vs. CNN(k=overparam).
    - Report p-values; p < 0.05 confirms statistical significance.
    - Summarize results in a table:

| Model                    | Parameter Count | Test Accuracy (%) | Test F1-Score (Macro) | McNemar's p-value (vs. SVM) |
|--------------------------|----------------|-------------------|-----------------------|------------------------------|
| SVM + Fisher Vectors     | ~[value]       | [value]           | [value]               | -                            |
| CNN(k=small)             | [value]        | [value]           | [value]               | [value]                      |
| CNN(k=critical)          | [value]        | [value]           | [value]               | [value]                      |
| CNN(k=overparam)         | [value]        | [value]           | [value]               | [value]                      |

**4.3. Analysis 2: Detailed Error Analysis (Rubric Criterion 2)**

- Generate and analyze full 10x10 confusion matrices for:
    - SVM Baseline
    - CNN(k=critical) (error peak)
    - CNN(k=overparam) (second descent)
- Summarize findings in a table:

| Error Pair (True → Pred)    | Error Rate (SVM) | Error Rate (CNN(k=critical)) | Error Rate (CNN(k=overparam)) |
|-----------------------------|------------------|------------------------------|-------------------------------|
| Cat → Dog                   | e.g., 30.1%      | e.g., 22.5%                  | e.g., 12.3%                   |
| Dog → Cat                   | e.g., 28.4%      | e.g., 20.1%                  | e.g., 10.5%                   |
| Truck → Automobile          | e.g., 20.5%      | e.g., 15.2%                  | e.g., 8.1%                    |
| Bird → Airplane             | e.g., 25.0%      | e.g., 18.3%                  | e.g., 9.7%                    |

This provides concrete evidence for the "implicit regularization" of the over-parameterized model. The analysis should argue that the CNN(k=critical) at the overfitting peak learns only coarse features (e.g., "is a 4-legged animal"), causing high cat/dog confusion, while CNN(k=overparam), though with more capacity to memorize, is guided by SGD toward a "smoother" solution learning finer-grained, more robust features (e.g., "snout shape", "ear shape"), thus resolving inter-class confusion.

---

V. Actionable Task List and 6-8 Page Report Structure

This final section presents an implementation plan fitting within project constraints.

**Implementation and Experimentation Task List:**

- *Code (Setup):*
    - Finalize dataset pipeline: 40k noisy train / 10k clean val / 10k clean test.
    - Implement SVM + Fisher Vector baseline.
    - Implement ScalableCNN(k) architecture.

- *Experiment (Run):*
    - Tune SVM baseline on validation set.
    - Tune ScalableCNN(k=8) model on validation set.
    - Lock all hyperparameters.
    - Train on all k values in {1...64} (computationally expensive—use Google Colab or similar).

- *Analysis (Generate Results):*
    - Generate the "Double Descent" plot.
    - Run McNemar's tests for Pair A and B.
    - Generate three confusion matrices.
    - Fill in Table 1 (quantitative results) and Table 2 (error analysis).

- *Writing (Report):*
    - Draft the full 6-8 page report, structured as follows:

**Proposed 6-8 Page Report Structure:**
1. **Introduction (1.5 pages):**
    - Use narrative from Section II (Classical Paradigm, Modern Paradox, Double Descent Hypothesis, RQs).
2. **Methodology (2 pages):**
    - Dataset & Preprocessing (20% label noise).
    - Experiment 1: Classical Baseline (SVM+FV).
    - Experiment 2: Scalable CNN Architecture.
    - Training and Tuning Details.
3. **Results and Analysis (3 pages):**
    - Baseline vs. Optimized CNN Performance (RQ1).
    - Inducing the Double Descent Curve (RQ2).
        - Present Figure 1 (Test Error vs. Model Width).
        - Discuss interpolation peak and "second descent."
    - Statistical and Error Analysis (RQ3).
        - Present Table 1 (Quantitative Performance).
        - McNemar's test results (criterion 1).
        - Present Table 2 (Qualitative Error Analysis).
        - Discuss error-pattern insights (criterion 2).
4. **Limitations (0.5 pages):** 
    - Only one architecture family tested; only one dataset (CIFAR-10); label noise was artificial.
5. **Conclusion (0.5 pages):**
    - Summarize findings and provide direct answers to the three RQs.
6. **References (0.5 pages):**
    - Cite all relevant literature.
