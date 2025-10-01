#!/usr/bin/env python3
import itertools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPTS = ['train_lr.py','train_dt.py','train_svm.py','train_knn.py','train_rf.py','train_gb.py']
SAMPLINGS = ['none','under','smote','ros']


def run_job(script: str, sampling: str) -> tuple[str, str, int]:
    module = os.path.splitext(script)[0]
    cmd = [
        'uv',
        'run',
        'python',
        '-c',
        (
            "import sys; "
            f"sys.path.insert(0, '{PROJECT_ROOT}'); "
            f"import models.{module} as mod; "
            f"mod.run('{sampling}')"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = 'ok' if result.returncode == 0 else 'fail'
    if status == 'fail':
        message = result.stderr.strip() or result.stdout.strip()
    else:
        message = result.stdout.strip()
    return f"{module}[{sampling}]", message, result.returncode


def main() -> None:
    jobs = list(itertools.product(SCRIPTS, SAMPLINGS))
    with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as executor:
        futures = {executor.submit(run_job, script, sampling): (script, sampling) for script, sampling in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Training jobs'):
            job_id, message, code = future.result()
            if code != 0:
                print(f"[ERROR] {job_id}: {message}")
            elif message:
                print(message)


if __name__ == '__main__':
    main()
