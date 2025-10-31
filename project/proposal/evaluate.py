import numpy as np
import os
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from shader import compute_points, compute_intensities, compute_global_bbox
from model import ShaderMLP
import time

# Setup output directories
os.makedirs('.cache/frames', exist_ok=True)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("-" * 50)

# Load data
print("Loading dataset...")
data = torch.load('.cache/ds/shader_dataset.pt', map_location='cpu')
x, y, t, intensity = data['x'], data['y'], data['t'], data['intensity']
inputs = torch.stack([x, y, t], dim=1).float()  # (N, 3)
targets = intensity.reshape(-1, 1).float()      # (N, 1)
print(f"Dataset shape: inputs={inputs.shape}, targets={targets.shape}")

# Dataset & split (same as training) - use the same seed by setting a consistent seed
print("Creating datasets...")
torch.manual_seed(42)  # Match training random seed
total_size = len(inputs)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Create same split as training
rand_idx = torch.randperm(total_size)
train_idx = rand_idx[:train_size]
val_idx = rand_idx[train_size:train_size+val_size]
test_idx = rand_idx[train_size+val_size:]

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Move to GPU
inputs = inputs.to(device)
targets = targets.to(device)

# Load trained model
print("Loading trained model...")
model = ShaderMLP().to(device)
model_path = '.cache/models/best_model.pth'
checkpoint = torch.load(model_path, map_location=device)

# Extract model state dict from checkpoint
state_dict = checkpoint['model_state_dict']

# Handle both compiled (_orig_mod) and non-compiled models
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    # Remove _orig_mod prefix if present
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from {model_path}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 50)

# Eval on test (MSE + PSNR on reconstructed frames)
print("Evaluating on test set...")
eval_start = time.time()

# Process test set in batches
batch_size = 8192  # Match training batch size
test_preds = []
test_trues = []
test_xy = []  # store (x, y) for test samples to enable bbox masking
with torch.no_grad():
    for i in range(0, len(test_idx), batch_size):
        batch_indices = test_idx[i:i+batch_size]
        batch_x = inputs[batch_indices]
        batch_y = targets[batch_indices]
        
        pred = model.predict(batch_x)
        test_preds.append(pred.cpu().numpy())
        test_trues.append(batch_y.cpu().numpy())
        test_xy.append(batch_x[:, :2].cpu().numpy())
test_preds = np.concatenate(test_preds)
test_trues = np.concatenate(test_trues)
test_xy = np.concatenate(test_xy)
mse = mean_squared_error(test_trues, test_preds)
psnr_val = psnr(test_trues, test_preds, data_range=1.0)
eval_time = time.time() - eval_start
print(f"Evaluation completed in {eval_time:.2f}s")
print(f"Test MSE: {mse:.6f}, PSNR: {psnr_val:.2f} dB")
 
# Compute PSNR within automatic bounding box (same logic as animations)
bmin_x, bmax_x, bmin_y, bmax_y = compute_global_bbox(device=device.type)
in_bbox = (
    (test_xy[:, 0] >= bmin_x) & (test_xy[:, 0] <= bmax_x) &
    (test_xy[:, 1] >= bmin_y) & (test_xy[:, 1] <= bmax_y)
)
num_bbox = int(in_bbox.sum())
if num_bbox > 0:
    mse_bbox = mean_squared_error(test_trues[in_bbox], test_preds[in_bbox])
    psnr_bbox = psnr(test_trues[in_bbox], test_preds[in_bbox], data_range=1.0)
    print(f"BBox [{bmin_x:.3f},{bmax_x:.3f}]x[{bmin_y:.3f},{bmax_y:.3f}] → {num_bbox} samples")
    print(f"BBox MSE: {mse_bbox:.6f}, PSNR: {psnr_bbox:.2f} dB")
else:
    print(f"BBox [{bmin_x:.3f},{bmax_x:.3f}]x[{bmin_y:.3f},{bmax_y:.3f}] selected 0 samples; skipping bbox PSNR")
print("-" * 50)

# Speed benchmark: Full 400x400 frame at sample t
print("Running inference benchmark...")
W = 400
xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, W))
grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
test_t = 0.5  # Mid-cycle (normalized)
test_grid = np.stack([xx.ravel(), yy.ravel(), np.full(W*W, test_t)], axis=1).astype(np.float32)
test_tensor = torch.tensor(test_grid).to(device)

# MLP inference
start = time.time()
with torch.no_grad():
    recon = model.predict(test_tensor).cpu().numpy().reshape(W, W)
mlp_time = (time.time() - start) * 1000  # ms
mlp_fps = 1000 / mlp_time
print(f"MLP Inference: {mlp_time:.1f} ms/frame → ~{mlp_fps:.0f} FPS (400x400) on {device}")

# Original shader computation (PyTorch)
print("Running shader benchmark (PyTorch)...")
actual_t = test_t * 12.5  # Convert normalized t to actual time
grid_xy_torch = torch.tensor(grid_xy, dtype=torch.float32).to(device)

# Warmup run to account for torch.compile
px, py = compute_points(actual_t, device=device)
points_norm_torch = torch.stack([px, py], dim=1)
_ = compute_intensities(points_norm_torch, grid_xy_torch, device=device)
if device == 'cuda':
    torch.cuda.synchronize()

# Actual benchmark
start = time.time()
for _ in range(10):  # Run 10 times to get average
    px, py = compute_points(actual_t, device=device)
    points_norm_torch = torch.stack([px, py], dim=1)
    intensities = compute_intensities(points_norm_torch, grid_xy_torch, device=device)
if device == 'cuda':
    torch.cuda.synchronize()
shader_time = ((time.time() - start) / 10) * 1000  # ms (averaged)
shader_fps = 1000 / shader_time
print(f"Shader: {shader_time:.1f} ms/frame → ~{shader_fps:.0f} FPS (400x400)")
shader_frame = intensities.cpu().numpy().reshape(W, W)

print("\nSpeed comparison:")
print(f"  Shader (compiled): {shader_time:.1f} ms/frame ({shader_fps:.1f} FPS)")
print(f"  MLP Model: {mlp_time:.1f} ms/frame ({mlp_fps:.1f} FPS)")
if shader_time < mlp_time:
    print(f"  Shader is {mlp_time/shader_time:.2f}x faster")
else:
    print(f"  MLP is {shader_time/mlp_time:.2f}x faster")
print("-" * 50)

# Visualize comparison: MLP vs Original Shader
print("Generating visualization...")
viz_start = time.time()
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(recon, cmap='gray', vmin=0, vmax=1)
plt.title(f'MLP Reconstruction (t={test_t:.2f})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(shader_frame, cmap='gray', vmin=0, vmax=1)
plt.title(f'Shader (t={test_t:.2f})')
plt.axis('off')

plt.subplot(1, 3, 3)
error_frame = np.abs(recon - shader_frame)
plt.imshow(error_frame, cmap='hot', vmin=0, vmax=0.1)
plt.title('Difference')
plt.axis('off')
plt.colorbar()

output_path = '.cache/frames/recon_sample.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
viz_time = time.time() - viz_start
print(f"Visualization saved to {output_path} in {viz_time:.2f}s")
print("-" * 50)

