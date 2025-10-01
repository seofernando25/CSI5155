#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / 'outputs'


def load_runs(model_name):
    rows = []
    model_dir = OUT_DIR / model_name
    for sampling in sorted(model_dir.iterdir()):
        metrics_path = sampling / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            rows.append({'sampling': sampling.name, **metrics})
    return pd.DataFrame(rows)


def display_confusion(model_name, sampling):
    import numpy as np
    metrics = json.load(open(OUT_DIR / model_name / sampling / 'metrics.json'))
    cm = metrics['confusion_matrix']
    plt.imshow(cm, cmap='Blues')
    plt.title(f'{model_name} ({sampling}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center', color='black')
    plt.show()

