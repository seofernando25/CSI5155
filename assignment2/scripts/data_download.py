import argparse
import zipfile
from urllib.request import urlretrieve
from lib.data_utils import DATASET_URL, DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download dataset")
    args = parser.parse_args()

    zip_path = DATA_DIR / DATASET_URL.split("/")[-1]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if zip_path.exists() and not args.force:
        print(f"Dataset already exists at {DATA_DIR}. Use --force to re-download.")
        return 0

    print(f"Downloading to {DATA_DIR}...")
    urlretrieve(DATASET_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()


if __name__ == "__main__":
    main()
