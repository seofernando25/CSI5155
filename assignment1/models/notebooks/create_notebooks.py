#!/usr/bin/env python3
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

MODELS = ['lr', 'dt', 'svm', 'knn', 'rf', 'gb']
SAMPLINGS = ['none', 'under', 'smote']

IMPORT_TEMPLATE = dedent(
    """
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    from models.train_{MODEL} import run
    from models.utils import load_dataframe, split_features
    """
)

CONFIG_TEMPLATE = dedent(
    """
    MODEL = '{MODEL}'
    SAMPLINGS = ['none', 'under', 'smote']
    """
)

RUN_TEMPLATE = dedent(
    """
    results = {}
    for samp in SAMPLINGS:
        print('Running sampling:', samp)
        results[samp] = run(samp)
    """
)

METRICS_TEMPLATE = dedent(
    """
    metrics_df = pd.DataFrame(
        [{'sampling': samp, **metrics} for samp, metrics in results.items()]
    )
    metrics_df[['sampling', 'precision', 'recall', 'roc_auc']]
    """
)

CONFUSION_TEMPLATE = dedent(
    """
    for samp, metrics in results.items():
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"{MODEL.upper()} - {samp}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                plt.text(j, i, cm[i][j], ha='center', va='center', color='black')
        plt.show()
    """
)

nb_dir = Path(__file__).resolve().parent
project_root = nb_dir.parents[1]

for model in MODELS:
    nb = nbf.v4.new_notebook()
    nb.cells.append(
        nbf.v4.new_markdown_cell(f"# {model.upper()} Model – Sampling Comparison")
    )
    nb.cells.append(
        nbf.v4.new_code_cell(IMPORT_TEMPLATE.replace('{MODEL}', model))
    )
    nb.cells.append(
        nbf.v4.new_code_cell(CONFIG_TEMPLATE.replace('{MODEL}', model))
    )
    nb.cells.append(nbf.v4.new_code_cell(RUN_TEMPLATE))
    nb.cells.append(nbf.v4.new_code_cell(METRICS_TEMPLATE))
    nb.cells.append(nbf.v4.new_code_cell(CONFUSION_TEMPLATE))

    nb_path = project_root / f"{model}.ipynb"
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
    print('Wrote notebook', nb_path)
