## Assignment 1

### Cmds

#### Tuning

```bash
uv run python tuning.py
uv run python tuning.py --model rf --sampling smote
```

#### 5-Fold Eval

```bash
uv run python five_fold_eval.py
```

#### Gen figs

```bash
uv run python generate_figures.py
```

#### Gen figs individually
```bash
uv run python -m figure_scripts.generate_roc_curves
uv run python -m figure_scripts.generate_confusion_matrices
uv run python -m figure_scripts.generate_dataset_analysis
uv run python -m figure_scripts.generate_engineered_features_analysis
```

