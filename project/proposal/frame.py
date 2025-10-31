import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shader import compute_points, compute_intensities, compute_global_bbox, T_MAX
from model import ShaderMLP

# Setup output directory
os.makedirs('.cache/frames', exist_ok=True)

GRID_RES = 256
TIME_VALUES = [0, 3.125, 6.25, 9.375, 12.5]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create grid in normalized space [-1, 1] for the whole canvas using PyTorch
print("Setting up grid...")
xx = torch.linspace(-1.0, 1.0, GRID_RES)
yy = torch.linspace(-1.0, 1.0, GRID_RES)
xx_grid, yy_grid = torch.meshgrid(xx, yy, indexing='xy')
grid_xy = torch.stack([xx_grid.ravel(), yy_grid.ravel()], dim=1).to(device)

# Get bounding box for visualization
bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = compute_global_bbox()
print(f"Animation bbox: [{bbox_min_x:.3f}, {bbox_max_x:.3f}] x [{bbox_min_y:.3f}, {bbox_max_y:.3f}]")

# Compute ground truth frames (shader)
shader_frames = []
for t in TIME_VALUES:
    print(f"GT shader frame at t={t:.2f}")
    px_norm, py_norm = compute_points(t, device=device)
    points_norm = torch.stack([px_norm, py_norm], dim=1)
    intensities = compute_intensities(points_norm, grid_xy, device=device)
    frame = intensities.cpu().numpy().reshape(GRID_RES, GRID_RES)
    shader_frames.append(frame)

# Load dataset for time-sample scatters
print("Loading dataset for time-sample row...")
data = torch.load('.cache/ds/shader_dataset.pt', map_location='cpu')
x_norm = data['x'].numpy()
y_norm = data['y'].numpy()
t_norm = data['t'].numpy()  # normalized [0,1]
intensity = data['intensity'].numpy()

# Define time windows matching plot_time_samples.py
time_ranges = [
    (0.0, 0.1),
    (3.125, 3.225),
    (6.250, 6.350),
    (9.375, 9.475),
    (12.4, 12.5)
]

# Prepare MLP model for inference row
print("Loading trained model for MLP row...")
mlp_model = ShaderMLP().to(device)
model_path = '.cache/models/best_model.pth'
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['model_state_dict']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

# Prepare grid for MLP inputs (x,y,t_norm)
xx_np, yy_np = np.meshgrid(np.linspace(-1.0, 1.0, GRID_RES), np.linspace(-1.0, 1.0, GRID_RES))
grid_xy_np = np.stack([xx_np.ravel(), yy_np.ravel()], axis=1).astype(np.float32)

mlp_frames = []
with torch.no_grad():
    for t_sec in TIME_VALUES:
        t_norm_val = np.float32(t_sec / T_MAX)
        inputs_np = np.column_stack([grid_xy_np, np.full(GRID_RES * GRID_RES, t_norm_val, dtype=np.float32)])
        inputs_t = torch.tensor(inputs_np).to(device)
        pred = mlp_model.predict(inputs_t).cpu().numpy().reshape(GRID_RES, GRID_RES)
        mlp_frames.append(pred)

# Build 3x5 figure
fig, axes = plt.subplots(3, len(TIME_VALUES), figsize=(20, 12))
fig.suptitle('Ground Truth vs Time Samples vs MLP Inference', fontsize=18)

# Row 1: Ground truth (shader)
for col, (t, frame) in enumerate(zip(TIME_VALUES, shader_frames)):
    ax = axes[0, col]
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1, extent=[-1.0, 1.0, -1.0, 1.0], origin='lower')
    rect = Rectangle((bbox_min_x, bbox_min_y), 
                     bbox_max_x - bbox_min_x, 
                     bbox_max_y - bbox_min_y,
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f't={t:.3f}s (GT)')
    ax.set_xlabel('x')
    if col == 0:
        ax.set_ylabel('Ground Truth')

# Row 2: Time samples (dataset)
for col, (start_t, end_t) in enumerate(time_ranges):
    ax = axes[1, col]
    ax.set_facecolor('red')
    start_norm = start_t / T_MAX
    end_norm = end_t / T_MAX
    mask = (t_norm > start_norm) & (t_norm < end_norm)
    if np.any(mask):
        ax.scatter(x_norm[mask], y_norm[mask], c=intensity[mask], cmap='gray', s=0.1, alpha=0.8, vmin=0, vmax=1)
        ax.set_title(f't=[{start_t:.3f},{end_t:.3f}]\n{int(mask.sum())} samples')
    else:
        ax.text(0.5, 0.5, f'No samples\n[{start_t:.3f},{end_t:.3f}]', ha='center', va='center', transform=ax.transAxes, color='white')
        ax.set_title(f't=[{start_t:.3f},{end_t:.3f}] (empty)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    if col == 0:
        ax.set_ylabel('Time Samples')
    ax.grid(True, color='white', alpha=0.3, linewidth=0.5)

# Row 3: MLP inference
for col, (t, frame) in enumerate(zip(TIME_VALUES, mlp_frames)):
    ax = axes[2, col]
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1, extent=[-1.0, 1.0, -1.0, 1.0], origin='lower')
    rect = Rectangle((bbox_min_x, bbox_min_y), 
                     bbox_max_x - bbox_min_x, 
                     bbox_max_y - bbox_min_y,
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f't={t:.3f}s (MLP)')
    ax.set_xlabel('x')
    if col == 0:
        ax.set_ylabel('MLP Inference')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = '.cache/frames/frames_comparison.png'
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print(f"\nSaved comparison to {output_path}")
plt.close()

print("\nFrames comparison (3x5) generated successfully!")
