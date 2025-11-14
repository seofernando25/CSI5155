# CIFAR-AlexNet Summary

## Model Architecture (`alexnet/model.py`)
- Backbone inspired by AlexNet, scaled for 32×32 inputs.
- Feature extractor (`nn.Sequential`):
  - `Conv(3→32, kernel=5, stride=1, pad=2)` → `BatchNorm2d(32)` → `ReLU` → `MaxPool(2,2)` (down to 16×16).
  - `Conv(32→64, kernel=5)` → `BatchNorm2d(64)` → `ReLU` → `MaxPool(2,2)` (down to 8×8).
  - `Conv(64→128, kernel=3)` → `BatchNorm2d(128)` → `ReLU` → `Dropout2d(0.1)`.
  - `Conv(128→128, kernel=3)` → `BatchNorm2d(128)` → `ReLU` → `Dropout2d(0.1)`.
  - `Conv(128→96, kernel=3)` → `BatchNorm2d(96)` → `ReLU` → `MaxPool(2,2)` (down to 4×4).
  - `AdaptiveAvgPool2d(1)` (global average pooling to 1×1×96).
- Classifier head:
  - `Flatten` → `Dropout(0.5)` → `Linear(96 → num_classes)`.
- Total parameters: **≈387 k**, all trainable.
- Regularization: BatchNorm in every conv block, light spatial dropout (0.1) mid-network, 0.5 dropout before linear layer.

## Training Script (`alexnet/train.py`)
- Entry command: `uv run python -m alexnet.train [--args]`.
- Defaults:
  - `epochs = 500`
  - `batch_size = 128`
  - `lr = 0.075`
  - `weight_decay = 2e-3`
  - `momentum = 0.9`
  - `label_smoothing = 0.1`
  - Scheduler: `CosineAnnealingLR(T_max=epochs)`
  - Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- Dataset: CIFAR-10 loaded from `.cache/base_datasets/cifar10`.
  - Training split: first 40 000 images (80%).
  - Validation split: remaining 10 000 images (20%).
  - Optional data augmentation commented out (random crop + flip).
  - Dataloaders use `num_workers=8`, pinning and prefetch tuned per device.
- Optimization loop:
  - Cross-entropy with label smoothing.
  - SGD updates each batch, scheduler stepped per epoch.
  - TensorBoard logging: batch-wise and epoch scalars, parameter/gradient histograms, and learning-rate traces written to `.cache/tensorboard/training/alexnet/run_*`.
  - No external tracking server is required; launch TensorBoard with `uv run python -m tensorboard.main --logdir .cache/tensorboard`.
  - Best validation checkpoint saved to `.cache/models/alexnet_cifar.pth` with optimizer state and metrics.

## Practical Notes
- Launch TensorBoard with `uv run python -m tensorboard.main --logdir .cache/tensorboard` to inspect runs.
- To disable logging, run with `tensorboard` environment variables or delete the writer block in `alexnet/train.py`.
- For reproducibility, set `--device` explicitly; defaults to CUDA if available else CPU.
- Training duration: sensitive to hardware; with CUDA, expect a few hours for 500 epochs; on CPU substantially longer.
- To enable augmentation, uncomment `train_transform` and pass to `CIFAR10Dataset`.

## Inference Usage
- Load weights: `state = torch.load(model_path); model.load_state_dict(state['model_state_dict'])`.
- Switch to eval mode (`model.eval()`) and forward normalized tensors shaped `(B, 3, 32, 32)`.
- For consistent preprocessing, normalize inputs to `[0,1]` float32, matching training loader behavior.

## Evaluation Snapshot (`uv run python -m alexnet.eval`)
- Dataset: CIFAR-10 test split (10 000 images).
- Model checkpoint: trained for **496 epochs**; best validation accuracy **83.17 %** saved.
- Test performance:
  - Accuracy **83.18 %** (8 318 / 10 000 correct).
  - Loss **0.6407**.
  - Evaluation latency **0.90 s** total → **0.09 ms per sample**.
- Per-class recalls:
  - airplane 85.4 %
  - automobile 91.5 %
  - bird 73.4 %
  - cat 66.8 %
  - deer 81.7 %
  - dog 77.3 %
  - frog 88.2 %
  - horse 85.6 %
  - ship 91.6 %
  - truck 90.3 %

