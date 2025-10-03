#!/usr/bin/env python3
"""
General utility functions for model training and evaluation.

This module provides:
- Sampling strategies (SMOTE, undersampling, oversampling)
- Model evaluation and reporting functions
- Pipeline building utilities
"""

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Sampling strategy constants
SAMPLING_STRATEGIES = ('none', 'under', 'smote')


def get_sampler(sampling: str):
    """
    Get a sampler instance for handling class imbalance.
    
    Args:
        sampling: One of 'none', 'under', 'smote'
    
    Returns:
        Sampler instance or None
    """
    if sampling is None or sampling == 'none':
        return None
    if sampling == 'smote':
        return SMOTE(random_state=42)
    if sampling == 'under':
        return RandomUnderSampler(random_state=42)
    raise ValueError(f"Unknown sampling: {sampling}")


def build_pipeline(preprocessor: ColumnTransformer, 
                   classifier, 
                   sampling: str = 'none') -> ImbPipeline:
    """
    Build a complete ML pipeline with preprocessing, sampling, and classification.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        classifier: Scikit-learn compatible classifier
        sampling: Sampling strategy to use
    
    Returns:
        Pipeline with all steps
    """
    sampler = get_sampler(sampling)
    steps = [('pre', preprocessor)]
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('clf', classifier))
    return ImbPipeline(steps)


