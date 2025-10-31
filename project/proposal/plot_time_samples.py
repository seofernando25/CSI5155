import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from shader import T_MAX

# Setup output directory
os.makedirs('.cache/frames', exist_ok=True)

# Load dataset
print("Loading dataset...")
data = torch.load('.cache/ds/shader_dataset.pt', map_location='cpu')
x_norm = data['x'].numpy()
y_norm = data['y'].numpy()
t = data['t'].numpy()
intensity = data['intensity'].numpy()
print(f"Dataset loaded: {len(x_norm)} samples")

# Define time ranges (in seconds) with 0.1 second windows
time_ranges = [
    (0.0, 0.1),
    (3.125, 3.225),
    (6.250, 6.350),
    (9.375, 9.475),
    (12.4, 12.5)  # Closer to T_MAX
]

# Create figure with subplots
fig, axes = plt.subplots(1, len(time_ranges), figsize=(20, 4))
fig.suptitle('Dataset Samples at Different Time Ranges', fontsize=16)

for i, (ax, (start_t, end_t)) in enumerate(zip(axes, time_ranges)):
    # Convert to normalized time
    start_t_sec = start_t / T_MAX
    end_t_sec = end_t / T_MAX
    
    # Filter samples in this time range
    mask = (t > start_t_sec) & (t < end_t_sec)
    x_filtered = x_norm[mask]
    y_filtered = y_norm[mask]
    int_filtered = intensity[mask]
    
    print(f"Time range [{start_t:.3f}s, {end_t:.3f}s]: {len(x_filtered)} samples")
    
    # Set red background
    ax.set_facecolor('red')
    
    if len(x_filtered) == 0:
        ax.text(0.5, 0.5, f'No samples\nin t=[{start_t:.3f}, {end_t:.3f}]', 
                ha='center', va='center', transform=ax.transAxes, color='white')
        ax.set_title(f't=[{start_t:.3f}s, {end_t:.3f}s] (empty)')
    else:
        # Plot scatter plot with intensity coloring
        ax.scatter(x_filtered, y_filtered, c=int_filtered, cmap='gray', s=1, alpha=0.8, vmin=0, vmax=1)
        ax.set_title(f't=[{start_t:.3f}s, {end_t:.3f}s]\n{len(x_filtered)} samples')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('y')
    ax.grid(True, color='white', alpha=0.3, linewidth=0.5)

plt.tight_layout()
output_path = '.cache/frames/time_samples_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {output_path}")
plt.close()

print("\nTime sample comparison completed!")
