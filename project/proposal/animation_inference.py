import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shader import T_MAX, compute_global_bbox
from matplotlib.patches import Rectangle
from model import ShaderMLP

# Setup output directory
os.makedirs('.cache/frames', exist_ok=True)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

GRID_RES = int(400)

# Animation settings
FPS = 12
DURATION = T_MAX/2  # seconds
TOTAL_FRAMES = int(FPS * DURATION)

# Get bounding box for visualization
bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = compute_global_bbox()
print(f"Animation bbox: [{bbox_min_x:.3f}, {bbox_max_x:.3f}] x [{bbox_min_y:.3f}, {bbox_max_y:.3f}]")

print("Setting up grid...")
xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, GRID_RES), np.linspace(-1.0, 1.0, GRID_RES))
grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
# Keep a torch copy on device to avoid per-frame transfers
grid_xy_t = torch.from_numpy(grid_xy.astype(np.float32)).to(device)

# Load the trained model
print("Loading trained model...")
model = ShaderMLP().to(device)
model_path = '.cache/models/best_model.pth'
checkpoint = torch.load(model_path, map_location=device)

# Extract the model state dict from checkpoint
state_dict = checkpoint['model_state_dict']

# Handle both compiled (_orig_mod) and non-compiled models
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    # Remove _orig_mod prefix if present
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from {model_path}")

print("Creating figure...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')
im = ax.imshow(
    np.zeros((GRID_RES, GRID_RES)),
    cmap='gray', vmin=0, vmax=1, animated=True,
    extent=[-1.0, 1.0, -1.0, 1.0], origin='lower'
)

# Draw red bounding box
rect = Rectangle((bbox_min_x, bbox_min_y), 
                 bbox_max_x - bbox_min_x, 
                 bbox_max_y - bbox_min_y,
                 linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

ax.axis('on')
ax.set_title('MLP Reconstruction Animation (t=0.00)')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame_num):
    t = (frame_num / TOTAL_FRAMES) * T_MAX  # Actual time for display
    t_normalized = frame_num / TOTAL_FRAMES  # Normalized time [0, 1] for model
    print(f"Frame {frame_num + 1}/{TOTAL_FRAMES}, t={t:.2f}")

    # Build inputs on device: (N, 3) with (x, y, t)
    with torch.no_grad():
        t_values_t = torch.full((grid_xy_t.shape[0], 1), float(t_normalized), device=device, dtype=torch.float32)
        input_tensor = torch.cat([grid_xy_t, t_values_t], dim=1)
        intensities_t = model.predict(input_tensor).view(GRID_RES, GRID_RES)

    # Convert only at the last step for plotting
    im.set_array(intensities_t.detach().cpu().numpy())
    ax.set_title(f'MLP Reconstruction Animation (t={t:.2f}, frame {frame_num + 1}/{TOTAL_FRAMES})')
    return [im]

print("Starting animation...")
anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, 
                                blit=True, repeat=False)

print("Saving video...")
output_path = '.cache/frames/mlp_inference_animation.mp4'
anim.save(output_path, fps=FPS, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved to {output_path}")

plt.show()
