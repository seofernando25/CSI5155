#!/usr/bin/env python3
"""Generate confusion matrix visualizations."""

import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


MODELS = {
    'lr': 'Logistic Regression',
    'dt': 'Decision Tree',
    'svm': 'SVM',
    'knn': 'k-NN',
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
}

SAMPLINGS = {
    'none': 'No Sampling',
    'under': 'Undersampling',
    'smote': 'SMOTE',
}


def generate_confusion_matrix_grid(output_dir='figures', tuning_file='tuning_results.json'):
    """Generate a grid of confusion matrices for all models and sampling strategies."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tuning results
    try:
        with open(tuning_file, 'r') as f:
            tuning_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {tuning_file} not found. Run tuning first.")
        return
    
    for sampling_key, sampling_label in SAMPLINGS.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (model_key, model_label) in enumerate(MODELS.items()):
            ax = axes[idx]
            
            # Find metrics for this model/sampling combination
            metrics = None
            for result in tuning_results:
                if result['model'] == model_key and result['sampling'] == sampling_key:
                    metrics = result['metrics']
                    break
            
            if metrics is not None:
                cm = np.array(metrics['confusion_matrix'])
                
                # Plot confusion matrix
                im = ax.imshow(cm, cmap='Blues')
                ax.set_title(f'{model_label}\n(P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f})',
                           fontsize=11, fontweight='bold')
                
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        text = ax.text(j, i, cm[i, j], ha='center', va='center',
                                     color='white' if cm[i, j] > cm.max()/2 else 'black',
                                     fontsize=14, fontweight='bold')
                
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['No Complaint', 'Complaint'])
                ax.set_yticklabels(['No Complaint', 'Complaint'])
            else:
                ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(model_label, fontsize=11, fontweight='bold')
        
        fig.suptitle(f'Confusion Matrices - {sampling_label}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'confusion_matrices_{sampling_key}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


if __name__ == '__main__':
    print("Generating confusion matrix figures...")
    generate_confusion_matrix_grid()
    print("\nAll confusion matrix figures generated successfully!")

