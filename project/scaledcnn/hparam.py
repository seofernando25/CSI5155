from pathlib import Path
import multiprocessing
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import optuna
import joblib
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scaledcnn.model import ScaledCNN


def objective(trial, train_dataset, val_dataset, device, k: int = 4, epochs: int = 500):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)

    # Enable cuDNN benchmark for faster convolutions
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = ScaledCNN(k=k, num_classes=10).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False
        ):
            optimizer.zero_grad()
            if use_amp:
                assert scaler is not None  # Type guard for mixed precision
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False
            ):
                if use_amp:
                    with autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def run(
    k: int = 4,
    epochs: int = 500,
    n_trials: int = 50,
    n_jobs: int = 1,
    device: str | None = None,
):
    torch_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {torch_device}")

    print("Pre-loading dataset into GPU memory...")
    print("Loading training data...")
    # Load directly from numpy files (much faster than PyArrow conversion)
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "processed_datasets" / "cifar10"

    train_images_np = np.load(dataset_path / "train_images.npy", mmap_mode="r")
    train_images = (
        torch.from_numpy(train_images_np.copy()).permute(0, 3, 1, 2).to(torch_device)
    )
    train_labels_np = np.load(dataset_path / "train_labels.npy", mmap_mode="r")
    train_labels = torch.from_numpy(train_labels_np.copy()).to(torch_device)
    print(
        f"Loaded {len(train_images)} training images to {torch_device} ({train_images.element_size() * train_images.nelement() / 1024**2:.1f} MB)"
    )

    print("Loading validation data...")
    val_images_np = np.load(dataset_path / "validation_images.npy", mmap_mode="r")
    val_images = (
        torch.from_numpy(val_images_np.copy()).permute(0, 3, 1, 2).to(torch_device)
    )
    val_labels_np = np.load(dataset_path / "validation_labels.npy", mmap_mode="r")
    val_labels = torch.from_numpy(val_labels_np.copy()).to(torch_device)
    print(
        f"Loaded {len(val_images)} validation images to {torch_device} ({val_images.element_size() * val_images.nelement() / 1024**2:.1f} MB)"
    )

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    print(f"\nInitializing ScaledCNN(k={k}) for hyperparameter tuning...")
    print(f"Will train for {epochs} epochs per trial")

    print("\n" + "=" * 60)
    print("Starting hyperparameter optimization with Optuna...")
    print("=" * 60)

    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Will distribute trials across GPUs.")
        # Pre-load data to each GPU
        train_datasets_per_gpu = {}
        val_datasets_per_gpu = {}
        for gpu_id in range(num_gpus):
            gpu_device = torch.device(f"cuda:{gpu_id}")
            print(f"Pre-loading data to GPU {gpu_id}...")
            train_images_gpu = train_images.to(gpu_device)
            train_labels_gpu = train_labels.to(gpu_device)
            val_images_gpu = val_images.to(gpu_device)
            val_labels_gpu = val_labels.to(gpu_device)
            train_datasets_per_gpu[gpu_id] = TensorDataset(
                train_images_gpu, train_labels_gpu
            )
            val_datasets_per_gpu[gpu_id] = TensorDataset(val_images_gpu, val_labels_gpu)
    else:
        train_datasets_per_gpu = {0: train_dataset}
        val_datasets_per_gpu = {0: val_dataset}

    study = optuna.create_study(
        direction="maximize",
        study_name=f"scaledcnn_k{k}_optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    def _objective(trial):
        # Assign GPU based on process ID (works with multiprocessing)
        if num_gpus > 1:
            # Use process ID to assign GPU consistently per process
            pid = multiprocessing.current_process().pid
            process_id = (pid if pid is not None else 0) % num_gpus
            device = torch.device(f"cuda:{process_id}")
            train_ds = train_datasets_per_gpu[process_id]
            val_ds = val_datasets_per_gpu[process_id]
        else:
            device = torch_device
            train_ds = train_dataset
            val_ds = val_dataset
        return objective(trial, train_ds, val_ds, device, k=k, epochs=epochs)

    start_time = time.time()
    study.optimize(
        _objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best hyperparameters: {study.best_params}")
    print(
        f"Best validation accuracy: {study.best_value:.4f} ({study.best_value * 100:.2f}%)"
    )
    print(f"Total optimization time: {elapsed / 60:.2f} minutes")
    print(f"Number of trials: {len(study.trials)}")
    print("\nTop 5 trials:")
    sorted_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True
    )
    for i, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is not None:
            print(
                f"  {i}. Accuracy: {trial.value:.4f} ({trial.value * 100:.2f}%) | "
                f"lr={trial.params.get('lr', 'N/A'):.6f}, "
                f"weight_decay={trial.params.get('weight_decay', 'N/A'):.6f}, "
                f"momentum={trial.params.get('momentum', 'N/A'):.4f}"
            )
    print("=" * 60)

    results_path = Path(f".cache/scaledcnn_hparam_k{k}_results.pkl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": study.trials,
            "k": k,
            "epochs": epochs,
            "n_trials": n_trials,
        },
        str(results_path),
    )
    print(f"\nResults saved to: {results_path}")

    return study.best_params, study.best_value


def add_subparser(subparsers):
    parser = subparsers.add_parser("hparam", help="Hyperparameter tuning with Optuna")
    parser.add_argument(
        "--k", type=int, default=4, help="Scaling factor k (fixed, default: 4)"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Epochs per trial (default: 500)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=250,
        help="Number of Optuna trials (default: 250)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, use 2-4 for GPU)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, default: auto)"
    )

    def _entry(args):
        return run(
            k=args.k,
            epochs=args.epochs,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            device=args.device,
        )

    parser.set_defaults(entry=_entry)
    return parser
