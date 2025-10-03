#!/usr/bin/env python3
"""Run hyperparameter tuning for all models with all sampling strategies."""

import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


MODELS = ['lr', 'dt', 'svm', 'knn', 'rf', 'gb']
SAMPLINGS = ['none', 'under', 'smote']


def run_tuning_job(model: str, sampling: str) -> tuple[str, str, int]:
    """Run a single tuning job."""
    cmd = ['uv', 'run', 'python', 'tune_model.py', '--model', model, '--sampling', sampling]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = 'ok' if result.returncode == 0 else 'fail'
    
    if status == 'fail':
        message = result.stderr.strip() or result.stdout.strip()
    else:
        # Extract key info from output
        lines = result.stdout.strip().split('\n')
        key_lines = [l for l in lines if 'Best params' in l or 'precision=' in l]
        message = '\n'.join(key_lines) if key_lines else 'Success'
    
    return f"{model}[{sampling}]", message, result.returncode


def main(max_workers: int = 4):
    """Run all tuning jobs in parallel."""
    jobs = list(itertools.product(MODELS, SAMPLINGS))
    
    print(f"Running {len(jobs)} tuning jobs with {max_workers} workers...")
    print("=" * 80)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_tuning_job, model, sampling): (model, sampling)
            for model, sampling in jobs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Tuning'):
            job_id, message, code = future.result()
            
            if code != 0:
                print(f"\n[ERROR] {job_id}:")
                print(message)
            else:
                print(f"\n[OK] {job_id}")
                if message:
                    print(message)
    
    print("\n" + "=" * 80)
    print("All tuning jobs completed!")
    print("\nRun 'python collect_results.py' to generate summary.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    args = parser.parse_args()
    
    main(max_workers=args.workers)

