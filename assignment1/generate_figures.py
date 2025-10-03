#!/usr/bin/env python3
"""
Generate all figures for the assignment.

This script runs all figure generation scripts in the figure_scripts/ folder.
"""

import os
import subprocess
import sys


def run_script(script_path: str):
    """Run a Python script and capture output."""
    print(f"\n{'=' * 80}")
    print(f"Running: {script_path}")
    print('=' * 80)
    
    # Run the script as a subprocess
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\nWarning: {script_path} exited with code {result.returncode}")


def main():
    """Generate all figures."""
    print("=" * 80)
    print("GENERATING ALL FIGURES FOR ASSIGNMENT")
    print("=" * 80)
    
    # List of figure generation scripts
    scripts = [
        'figure_scripts/generate_roc_curves.py',
        'figure_scripts/generate_confusion_matrices.py',
    ]
    
    for script in scripts:
        if os.path.exists(script):
            try:
                run_script(script)
            except Exception as e:
                print(f"\nError running {script}: {e}")
        else:
            print(f"\nWarning: {script} not found, skipping...")
    
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated figures are saved in the 'figures/' directory.")
    print("Check the following files:")
    print("  - figures/roc_curves_all_models.png")
    print("  - figures/roc_curves_none.png")
    print("  - figures/roc_curves_under.png")
    print("  - figures/roc_curves_smote.png")
    print("  - figures/roc_curves_ros.png")
    print("  - figures/confusion_matrices_none.png")
    print("  - figures/confusion_matrices_under.png")
    print("  - figures/confusion_matrices_smote.png")
    print("  - figures/confusion_matrices_ros.png")


if __name__ == '__main__':
    main()

