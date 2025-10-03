#!/usr/bin/env python3
"""Collect results from tuning_results.json and create a summary CSV."""

import json
import pandas as pd


def collect_results(input_file='tuning_results.json', output_file='summary.csv'):
    """
    Collect all metrics from tuning results and create a summary DataFrame.
    
    Args:
        input_file: JSON file with tuning results
        output_file: CSV file to save summary
    """
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run tuning first.")
        return None
    
    # Extract rows for DataFrame
    rows = []
    for result in results:
        rows.append({
            'model': result['model'],
            'sampling': result['sampling'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall'],
            'roc_auc': result['metrics']['roc_auc'],
        })
    
    df = pd.DataFrame(rows).sort_values(['model', 'sampling'])
    
    # Save summary
    df.to_csv(output_file, index=False)
    
    print("Summary of all model results:")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_file}")
    
    return df


if __name__ == '__main__':
    collect_results()

