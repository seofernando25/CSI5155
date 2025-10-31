import numpy as np
import os
import matplotlib.pyplot as plt
from shader import T_MAX

# Setup output directory
os.makedirs('.cache/frames', exist_ok=True)

# Load dataset
data = np.load('.cache/ds/shader_dataset.npz')
x_norm, y_norm, t, intensity = data['x'], data['y'], data['t'], data['intensity']


# Time seconds
start_t = 2
end_t = 2.1

# Convert to normalized time
start_t_sec = start_t / T_MAX
end_t_sec = end_t / T_MAX

duration_sec = end_t_sec - start_t_sec

mask = (t > start_t_sec) & (t < end_t_sec) 
x_filtered = x_norm[mask]
y_filtered = y_norm[mask]
int_filtered = intensity[mask]

print(f"Time range: t={start_t}s to {end_t}s (normalized: {start_t_sec:.4f} to {end_t_sec:.4f})")
print(f"Filtered samples: {len(x_filtered)} samples in filtered range")

assert len(int_filtered) > 0, f"No samples found in time range t=[{start_t}s, {end_t}s]"
print(f"Sample intensities: {int_filtered[:5]}")
print(f"Sample normalized coordinates: x={x_filtered[:5]}, y={y_filtered[:5]}")

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('red')  # Red background

# Point size for scatter plot
ax.scatter(x_filtered, y_filtered, c=int_filtered, cmap='gray', s=1, alpha=0.8)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title(f'Dataset samples t=[{start_t:.1f}s, {end_t:.1f}s]')
ax.grid(True, color='white', alpha=0.3, linewidth=0.5)  # Add grid
ax.axis('on')  # Show axes for the grid
output_path = f'.cache/frames/sanity_check_t{start_t:.2f}_t{end_t:.2f}.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"Saved to {output_path}")
plt.show()

# Quick stats
print("\n=== Filtered Dataset Statistics ===")
print(f"Intensity range: {np.min(int_filtered):.4f} - {np.max(int_filtered):.4f}, mean: {np.mean(int_filtered):.4f}")
print(f"Spatial coverage: x range [{np.min(x_filtered):.4f}, {np.max(x_filtered):.4f}], y range [{np.min(y_filtered):.4f}, {np.max(y_filtered):.4f}]")
print(f"Unique x values: {len(np.unique(x_filtered))}, Unique y values: {len(np.unique(y_filtered))}")

# Spatial distribution analysis
print("\n=== Spatial Distribution Analysis ===")
print(f"Total points in filtered dataset: {len(x_filtered)}")
print(f"X range: [{np.min(x_filtered):.4f}, {np.max(x_filtered):.4f}]")
print(f"Y range: [{np.min(y_filtered):.4f}, {np.max(y_filtered):.4f}]")

print("\nX coordinate distribution:")
x_bins = np.linspace(-1, 1, 21)
x_counts, _ = np.histogram(x_filtered, bins=x_bins)
for i in range(len(x_bins)-1):
    if x_counts[i] > 0:
        print(f"  x=[{x_bins[i]:.2f},{x_bins[i+1]:.2f}]: {x_counts[i]:4d} points")