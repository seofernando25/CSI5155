import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATASET_URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
OUTPUT_DIR = Path(__file__).parent / "dataset"
ZIP_PATH = OUTPUT_DIR / "optical_recognition_of_handwritten_digits.zip"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path) -> None:
    # Stream download to avoid keeping the whole file in memory
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:
        # Some servers may not provide content-length
        chunk_size = 1024 * 1024  # 1 MiB
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def main() -> int:
    print(f"Preparing to download dataset to: {OUTPUT_DIR}")
    ensure_output_dir(OUTPUT_DIR)

    if ZIP_PATH.exists():
        print(f"Existing zip found at {ZIP_PATH}. Re-downloading to refresh...")
        try:
            ZIP_PATH.unlink()
        except OSError:
            print("Warning: could not remove existing zip; attempting to overwrite.")

    print(f"Downloading: {DATASET_URL}")
    try:
        download_file(DATASET_URL, ZIP_PATH)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return 1

    print("Download complete. Extracting...")
    try:
        extract_zip(ZIP_PATH, OUTPUT_DIR)
    except zipfile.BadZipFile:
        print("Downloaded file is not a valid zip archive.")
        return 1
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return 1

    print(f"Dataset extracted to: {OUTPUT_DIR}")
    # Keep the zip for reproducibility; comment out to delete
    # ZIP_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
