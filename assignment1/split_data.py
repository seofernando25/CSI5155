#!/usr/bin/env python3
"""
Data splitting script for marketing campaign dataset.
Splits data into train/validation/test sets with proper stratification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pipeline import load_dataframe
import os

def split_and_save_data():
    """Split data into train/validation/test sets and save to CSV files."""
    
    print("Loading marketing campaign data...")
    df = load_dataframe()
    
    # Separate features and target
    X = df.drop('Complain', axis=1)
    y = df['Complain']
    
    print(f"Total samples: {len(df)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Target ratio: {y.mean():.4f}")
    
    # First split: 80% train+val, 20% test
    print("\nSplitting into train+val (80%) and test (20%)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    print("Splitting train+val into train (75%) and val (25%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_train_val
    )
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save splits
    print("\nSaving data splits...")
    X_train.to_csv('data/train_features.csv', index=False)
    y_train.to_csv('data/train_target.csv', index=False)
    X_val.to_csv('data/val_features.csv', index=False)
    y_val.to_csv('data/val_target.csv', index=False)
    X_test.to_csv('data/test_features.csv', index=False)
    y_test.to_csv('data/test_target.csv', index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA SPLIT SUMMARY")
    print("="*50)
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  - Target ratio: {y_train.mean():.4f}")
    print(f"Val set:   {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  - Target ratio: {y_val.mean():.4f}")
    print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  - Target ratio: {y_test.mean():.4f}")
    print("\nFiles saved:")
    print("  - data/train_features.csv")
    print("  - data/train_target.csv")
    print("  - data/val_features.csv")
    print("  - data/val_target.csv")
    print("  - data/test_features.csv")
    print("  - data/test_target.csv")
    print("="*50)

if __name__ == '__main__':
    split_and_save_data()
