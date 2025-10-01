#!/usr/bin/env python3
import os
import json
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), 'outputs')
rows = []
for model in os.listdir(BASE):
    mdir = os.path.join(BASE, model)
    if not os.path.isdir(mdir):
        continue
    for sampling in os.listdir(mdir):
        sdir = os.path.join(mdir, sampling)
        if not os.path.isdir(sdir):
            continue
        mj = os.path.join(sdir, 'metrics.json')
        if os.path.exists(mj):
            with open(mj) as f:
                m = json.load(f)
                rows.append({
                    'model': model,
                    'sampling': sampling,
                    'precision': m.get('precision'),
                    'recall': m.get('recall'),
                    'roc_auc': m.get('roc_auc'),
                })

df = pd.DataFrame(rows).sort_values(['model','sampling'])
out_csv = os.path.join(BASE, 'summary.csv')
df.to_csv(out_csv, index=False)
print(df)
print('\nWrote', out_csv)
