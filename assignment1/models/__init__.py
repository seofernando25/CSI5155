"""
Clean model definitions with simple APIs.

Each model module exports:
- get_model(): Returns an initialized model instance
- get_param_grid(): Returns the hyperparameter grid for tuning
"""

from .logistic_regression import get_model as get_lr_model, get_param_grid as get_lr_param_grid
from .decision_tree import get_model as get_dt_model, get_param_grid as get_dt_param_grid
from .svm import get_model as get_svm_model, get_param_grid as get_svm_param_grid
from .knn import get_model as get_knn_model, get_param_grid as get_knn_param_grid
from .random_forest import get_model as get_rf_model, get_param_grid as get_rf_param_grid
from .gradient_boosting import get_model as get_gb_model, get_param_grid as get_gb_param_grid

__all__ = [
    'get_lr_model', 'get_lr_param_grid',
    'get_dt_model', 'get_dt_param_grid',
    'get_svm_model', 'get_svm_param_grid',
    'get_knn_model', 'get_knn_param_grid',
    'get_rf_model', 'get_rf_param_grid',
    'get_gb_model', 'get_gb_param_grid',
]

