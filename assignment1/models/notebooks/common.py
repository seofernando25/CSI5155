#!/usr/bin/env python3
import os
import json
from pathlib import Path

def load_metrics(model_name: str):
    base = Path(__file__).resolve().parent.parent / 'outputs' / model_name
    rows = []
    for sampling_dir in sorted(base.iterdir()):
        metrics_path = sampling_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            rows.append({'sampling': sampling_dir.name, **m})
    return rows

