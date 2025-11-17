import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image
import uvicorn
from utils import require_file
from data.datasets import get_cifar10_class_names


# Global variables to store dataset info
base_datasets_info: Dict[str, Any] = {}
processed_datasets_info: Dict[str, Any] = {}


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def pil_to_np(image_data):
    """Convert image data to numpy array, handling PIL Images, numpy arrays, and lists."""
    if isinstance(image_data, Image.Image):
        return np.array(image_data)
    elif isinstance(image_data, np.ndarray):
        return image_data
    elif isinstance(image_data, (list, tuple)):
        # Handle lists/tuples (which can occur with datasets library)
        return np.asarray(image_data)
    else:
        return image_data


def load_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    loaded = load_from_disk(str(dataset_path))
    assert isinstance(loaded, DatasetDict), "Expected DatasetDict"
    ds_dict: DatasetDict = loaded
    info = {
        "name": dataset_path.name,
        "splits": list(ds_dict.keys()),
        "total_samples": sum(len(ds) for ds in ds_dict.values()),
        "features": {},
    }

    first_split_name = list(ds_dict.keys())[0]
    ds = ds_dict[first_split_name]

    info["features"] = {name: str(feature) for name, feature in ds.features.items()}

    image_col = None
    if "image" in ds.features:
        image_col = "image"
    elif "img" in ds.features:
        image_col = "img"
    if image_col is None:
        raise ValueError("No image column found")

    # Get first image and convert to numpy array
    first_image = pil_to_np(ds[0][image_col])
    if isinstance(first_image, np.ndarray):
        info["dimensions"] = list(first_image.shape)
    else:
        # Fallback: try to get shape from the feature metadata
        info["dimensions"] = "Unknown"

    # Sample min and max over first up to 100 samples
    sample_images = [pil_to_np(ds[i][image_col]) for i in range(min(100, len(ds)))]
    # Ensure all are numpy arrays
    sample_images = [
        img if isinstance(img, np.ndarray) else np.asarray(img) for img in sample_images
    ]
    sample_images_np = np.stack(sample_images)
    info["min_value"] = make_serializable(np.min(sample_images_np))
    info["max_value"] = make_serializable(np.max(sample_images_np))
    return info


def image_to_base64(image_data) -> str:
    """Convert image data to base64 PNG string, handling normalized [0,1] and [0,255] ranges."""
    arr = pil_to_np(image_data)

    # Ensure it's a numpy array
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    # Handle normalized [0,1] range: convert to [0,255]
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            # Already in [0,1] range, scale to [0,255]
            arr = (arr * 255.0).astype(np.uint8)
        else:
            # In some other range, normalize to [0,255]
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
                np.uint8
            )
    elif arr.dtype != np.uint8:
        # Other integer types or non-uint8, normalize to [0,255]
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
            np.uint8
        )

    if arr.ndim == 2:
        pil_image = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_image = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global base_datasets_info, processed_datasets_info

    repo_root = Path(__file__).resolve().parents[1]

    base_root = repo_root / ".cache" / "base_datasets"
    assert base_root.exists(), "Base datasets not found"
    for dataset_dir in base_root.iterdir():
        if dataset_dir.is_dir():
            base_datasets_info[dataset_dir.name] = load_dataset_info(dataset_dir)

    processed_root = require_file(repo_root / ".cache" / "processed_datasets")
    for dataset_dir in processed_root.iterdir():
        if dataset_dir.is_dir():
            processed_datasets_info[dataset_dir.name] = load_dataset_info(dataset_dir)

    yield

    base_datasets_info.clear()
    processed_datasets_info.clear()


app = FastAPI(
    title="Dataset Explorer",
    description="Explore base and processed datasets",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = Path(__file__).resolve().parents[1] / "web" / "ds_explorer.html"
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/ds_explorer.css")
async def get_css():
    css_path = Path(__file__).resolve().parents[1] / "web" / "ds_explorer.css"
    with open(css_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")


@app.get("/api/datasets")
async def list_datasets():
    return {
        "base_datasets": base_datasets_info,
        "processed_datasets": processed_datasets_info,
    }


@app.get("/api/datasets/{dataset_type}")
async def get_dataset_info(
    dataset_type: Literal["base", "processed"],
):
    if dataset_type not in ["base", "processed"]:
        raise HTTPException(status_code=400, detail="Invalid dataset type")
    dataset_name = "cifar10"  # Only one dataset available
    return (
        base_datasets_info[dataset_name]
        if dataset_type == "base"
        else processed_datasets_info[dataset_name]
    )


@app.get("/api/datasets/{dataset_type}/{split}")
async def get_dataset_samples(
    dataset_type: Literal["base", "processed"],
    split: Literal["train", "test", "validation"],
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    assert dataset_type in ["base", "processed"], "Invalid dataset type"
    dataset_name = "cifar10"  # Only one dataset available
    assert (
        dataset_name in base_datasets_info or dataset_name in processed_datasets_info
    ), f"Dataset {dataset_name} in {dataset_type} not found"

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / f"{dataset_type}_datasets" / dataset_name

    loaded = load_from_disk(str(dataset_path))
    assert isinstance(loaded, DatasetDict), "Expected DatasetDict"
    ds_dict: DatasetDict = loaded

    if split not in ds_dict:
        raise HTTPException(status_code=404, detail="Split not found")

    ds: Dataset = ds_dict[split]

    image_col = None
    if "img" in ds.features:
        ds = ds.rename_column("img", "image")
        image_col = "image"
    elif "image" in ds.features:
        image_col = "image"

    total_samples = len(ds)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_samples)

    # Get class names for label mapping
    class_names = get_cifar10_class_names()

    samples = []
    for i in range(start_idx, end_idx):
        sample = ds[i]
        sample = make_serializable(sample)

        # Add label name if label exists
        if "label" in sample:
            label_idx = int(sample["label"])
            if 0 <= label_idx < len(class_names):
                sample["label_name"] = class_names[label_idx]

        if image_col and image_col in sample:
            sample["image_base64"] = image_to_base64(sample[image_col])
            del sample[image_col]

        samples.append(sample)

    return {
        "samples": samples,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_samples": total_samples,
            "total_pages": (total_samples + page_size - 1) // page_size,
            "has_next": end_idx < total_samples,
            "has_prev": page > 1,
        },
    }


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("data.explorer:app", host=host, port=port, log_level="info")
