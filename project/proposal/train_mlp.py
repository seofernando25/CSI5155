import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
from model import ShaderMLP, compute_spectral_centroid
from shader import compute_intensities, compute_points, T_MAX, W
from skimage.metrics import structural_similarity as ssim


# Setup output directories
os.makedirs('.cache/frames', exist_ok=True)
os.makedirs('.cache/models', exist_ok=True)
os.makedirs('.cache/logs', exist_ok=True)

# TensorBoard writer
writer = SummaryWriter('.cache/logs')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    torch.set_float32_matmul_precision('high')
print("-" * 50)

# Config: periodic checkpoint interval (epochs)
CHECKPOINT_EVERY = 1000
N_FRAME_ACCUMULATION = 5  # number of temporal frames per optimizer step

# Low-discrepancy sequence for times (golden-ratio conjugate)
GOLDEN_RATIO_CONJ = (5 ** 0.5 - 1) / 2  # ≈ 0.6180339887

def ld_time(epoch_index: int, offset: float = 0.0) -> float:
    """Returns a physical time in [0, T_MAX] using a low-discrepancy sequence.
    Uses fractional part of (offset + n * φ_conj) mapped to [0, T_MAX].
    """
    frac = (offset + epoch_index * GOLDEN_RATIO_CONJ) % 1.0
    return frac * T_MAX

# Utility: compute SSIM on a frame given predictions and ground truth
def compute_frame_ssim(pred_full: torch.Tensor, gt_full: torch.Tensor) -> float:
    """Compute SSIM between predicted and ground truth frames."""
    # SSIM expects numpy arrays
    pred_np = pred_full.detach().cpu().numpy()
    gt_np = gt_full.detach().cpu().numpy() if isinstance(gt_full, torch.Tensor) else gt_full
    score = ssim(gt_np, pred_np, data_range=1.0)
    return float(score)

print("Preparing on-the-fly training grid...")
GRID_RES = int(W)
# Full normalized grid on device (reused every epoch)
x_lin = torch.linspace(-1.0, 1.0, GRID_RES, device=device)
y_lin = torch.linspace(-1.0, 1.0, GRID_RES, device=device)
xx, yy = torch.meshgrid(x_lin, y_lin, indexing='xy')
grid_xy_full = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
num_grid_points = grid_xy_full.shape[0]
print(f"Grid prepared: {GRID_RES}x{GRID_RES} => {num_grid_points:,} points")

# Preallocate reusable buffers to minimize per-iteration allocations
# Full-frame model input [x, y, t]
inputs_full = torch.empty((num_grid_points, 3), device=device, dtype=torch.float32)
inputs_full[:, :2] = grid_xy_full  # static part
t_full_col = inputs_full[:, 2:3]   # view for fast fill_


# Compute spectral centroid on sample frame for WINNER
print("Computing spectral centroid for WINNER init...")
SAMPLE_T = 0.5
px_sample, py_sample = compute_points(SAMPLE_T, device=device.type)
points_sample = torch.stack([px_sample, py_sample], dim=1)
with torch.no_grad():
    gt_sample_flat = compute_intensities(points_sample, grid_xy_full, device=device.type)
    gt_sample_frame = gt_sample_flat.reshape(GRID_RES, GRID_RES).clamp(0.0, 1.0)
spectral_centroid = compute_spectral_centroid(gt_sample_frame)
print(f"Spectral centroid (ψ): {spectral_centroid:.2f}")

# Initialize model with WINNER
model = ShaderMLP(spectral_centroid=spectral_centroid).to(device)
# Use torch.compile for faster training
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
# Use class-imbalanced BCE with logits to avoid collapsing to black
criterion = nn.BCEWithLogitsLoss(reduction='mean')

# Log WINNER params
writer.add_scalar('WINNER/spectral_centroid', spectral_centroid, 0)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 50)

# Train
print("Starting training...")
epochs = 50_000
model.train()
train_start = time.time()

train_losses = []
val_losses = []
train_ssims = []
val_ssims = []

# Checkpointing variables (track best validation SSIM)
best_val_ssim = float('-inf')
checkpoint_path = '.cache/models/best_model.pth'

for epoch in range(epochs):
    epoch_start = time.time()
    # Separate CUDA graph steps if available (avoids output reuse issues)
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
        torch.compiler.cudagraph_mark_step_begin()
    
    # Training phase with gradient accumulation over N temporal frames for smoother updates
    optimizer.zero_grad()
    total_frame_gen_time = 0.0
    train_ssim_accum = 0.0
    avg_train_loss = 0.0
    # Generate N quasi-random times using golden-ratio step to avoid repeats for N>2
    t_trains = [ld_time(epoch, offset=(0.37 + k * GOLDEN_RATIO_CONJ) % 1.0) for k in range(N_FRAME_ACCUMULATION)]
    for t_train in t_trains:
        frame_gen_start = time.time()
        px_norm, py_norm = compute_points(t_train, device=device.type)
        points_norm = torch.stack([px_norm, py_norm], dim=1)
        with torch.no_grad():
            gt_flat = compute_intensities(points_norm, grid_xy_full, device=device.type).reshape(-1, 1).clone()
        total_frame_gen_time += (time.time() - frame_gen_start)
        t_full_col.fill_(float(t_train / T_MAX))
        logits_full = model.forward_logits(inputs_full)
        # Dynamic positive weight to counter sparsity (num_neg/num_pos)
        with torch.no_grad():
            pos = (gt_flat > 0.5).sum()
            neg = gt_flat.numel() - pos
            # Avoid divide-by-zero; if no positives, skip weighting (weight=1)
            pos_weight_val = (neg.float() / (pos.float() + 1e-6)).clamp(max=1000.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        loss = bce(logits_full, gt_flat)
        (loss / N_FRAME_ACCUMULATION).backward()
        avg_train_loss += loss.item() / N_FRAME_ACCUMULATION

        # Compute SSIM for this frame (evaluation only)
        model.eval()
        with torch.no_grad():
            pred_full = torch.sigmoid(logits_full).reshape(GRID_RES, GRID_RES).clamp(0.0, 1.0)
            gt_full = gt_flat.reshape(GRID_RES, GRID_RES).clamp(0.0, 1.0)
        model.train()
        train_ssim_accum += compute_frame_ssim(pred_full, gt_full) / N_FRAME_ACCUMULATION

    optimizer.step()

    frame_gen_time = total_frame_gen_time
    train_losses.append(avg_train_loss)
    train_ssim = train_ssim_accum
    train_ssims.append(train_ssim)
    
    # Validation phase
    model.eval()
    # Validation on a different low-discrepancy sequence (different offset to decorrelate)
    t_val = ld_time(epoch, offset=0.837)
    frame_gen_val_start = time.time()
    px_val, py_val = compute_points(t_val, device=device.type)
    points_val = torch.stack([px_val, py_val], dim=1)
    with torch.no_grad():
        gt_val_flat = compute_intensities(points_val, grid_xy_full, device=device.type).reshape(-1, 1).clone()
        t_full_col.fill_(float(t_val / T_MAX))
        logits_val_full = model.forward_logits(inputs_full)
        pred_val_full = torch.sigmoid(logits_val_full).reshape(GRID_RES, GRID_RES).clamp(0.0, 1.0)
        gt_val_full = gt_val_flat.reshape(GRID_RES, GRID_RES).clamp(0.0, 1.0)
    frame_gen_val_time = time.time() - frame_gen_val_start
    # Report BCEWithLogits on validation as well
    with torch.no_grad():
        pos = (gt_val_flat > 0.5).sum()
        neg = gt_val_flat.numel() - pos
        pos_weight_val = (neg.float() / (pos.float() + 1e-6)).clamp(max=1000.0)
        bce_val = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        avg_val_loss = bce_val(logits_val_full, gt_val_flat).item()
    val_losses.append(avg_val_loss)
    val_ssim = compute_frame_ssim(pred_val_full, gt_val_full)
    val_ssims.append(val_ssim)
    model.train()
    
    epoch_time = time.time() - epoch_start
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
    writer.add_scalar('SSIM/Train', train_ssim, epoch + 1)
    writer.add_scalar('SSIM/Validation', val_ssim, epoch + 1)
    writer.add_scalar('Time/Epoch', epoch_time, epoch + 1)
    writer.add_scalar('Time/FrameGenTrain', frame_gen_time, epoch + 1)
    writer.add_scalar('Time/FrameGenVal', frame_gen_val_time, epoch + 1)
    
    # Save checkpoint if best validation SSIM improved
    if val_ssim > best_val_ssim:
        best_val_ssim = val_ssim
        save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ssims': train_ssims,
            'val_ssims': val_ssims,
            'best_val_ssim': best_val_ssim
        }, checkpoint_path)
        print(f"✓ Checkpoint saved (best val SSIM: {best_val_ssim:.4f})")

    # Periodic checkpoint every CHECKPOINT_EVERY epochs
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        periodic_path = f".cache/models/ckpt_epoch_{epoch + 1:06d}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ssims': train_ssims,
            'val_ssims': val_ssims,
            'best_val_ssim': best_val_ssim
        }, periodic_path)
        print(f"✓ Periodic checkpoint saved: {periodic_path}")
    
    print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s), Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

total_train_time = time.time() - train_start
print(f"\nTraining completed in {total_train_time:.2f}s")

# Load best model
print("\nLoading best checkpoint (by validation SSIM)...")
checkpoint = torch.load(checkpoint_path)
save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
save_model.load_state_dict(checkpoint['model_state_dict'])
if 'best_val_ssim' in checkpoint:
    print(f"Best model loaded from epoch {checkpoint['epoch']} (val SSIM: {checkpoint['best_val_ssim']:.4f})")
else:
    print(f"Best model loaded from epoch {checkpoint['epoch']}")

# Plot training curves
print("Plotting training curves...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Loss plot
axes[0].plot(range(1, epochs + 1), train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
axes[0].plot(range(1, epochs + 1), val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (log scale)', fontsize=12)
axes[0].set_yscale('log')
axes[0].set_title('Training and Validation Loss', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, which='both')

# SSIM plot
axes[1].plot(range(1, epochs + 1), train_ssims, 'b-o', label='Train SSIM', linewidth=2, markersize=6)
axes[1].plot(range(1, epochs + 1), val_ssims, 'r-s', label='Validation SSIM', linewidth=2, markersize=6)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('SSIM', fontsize=12)
axes[1].set_title('Training and Validation SSIM', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = '.cache/frames/training_curves.png'
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Training curves saved to {loss_plot_path}")
print("-" * 50)

# Save final model (best checkpoint already saved)
print("\nBest model already saved at: .cache/models/best_model.pth")
if 'best_val_ssim' in checkpoint:
    print(f"Best validation SSIM: {checkpoint['best_val_ssim']:.4f}")
print("-" * 50)
print(f"Total execution time: {time.time() - train_start:.2f}s")

# Close TensorBoard writer
writer.close()
print("\nTensorBoard logs saved to: .cache/logs")
print("View with: tensorboard --logdir=.cache/logs")
print("-" * 50)
print("\nTraining complete! Run evaluate.py to evaluate the model on the test set.")