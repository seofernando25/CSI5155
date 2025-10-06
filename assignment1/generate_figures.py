import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os


def run_script(script_module: str) -> tuple[str, str, int]:
    result = subprocess.run(
        [sys.executable, "-m", script_module], capture_output=True, text=True
    )

    message = result.stderr.strip() if result.stderr.strip() else "Success"

    return script_module, message, result.returncode


def main():
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_script, script): script
            # Modules to generate figures
            for script in [
                "figure_scripts.generate_roc_curves",
                "figure_scripts.generate_confusion_matrices",
                "figure_scripts.generate_dataset_analysis",
                "figure_scripts.generate_engineered_features_analysis",
            ]
        }

        for future in tqdm(
            as_completed(futures), total=len(futures)
        ):
            module_name, message, code = future.result()

            if code != 0:
                print(f"\n[ERROR] {module_name}:")
                print(message)
            else:
                print(f"\n[OK] {module_name}")
                if message and message != "Success":
                    print(message)


if __name__ == "__main__":
    main()
