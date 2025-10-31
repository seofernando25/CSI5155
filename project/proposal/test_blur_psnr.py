"""
Test how PSNR changes with Gaussian blur strength.
Renders a 400x400 frame at t=0.5 and applies increasing blur.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from shader import compute_points, compute_intensities

# Setup
os.makedirs('.cache/frames', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Parameters
GRID_RES = 400
test_t = 0.5  # Normalized time
actual_t = test_t * 12.5  # Convert to actual time
print(f"Rendering frame at t={test_t} (actual t={actual_t:.3f})")

# Create grid in normalized space
print("Creating grid...")
xx = torch.linspace(-1.0, 1.0, GRID_RES)
yy = torch.linspace(-1.0, 1.0, GRID_RES)
xx_grid, yy_grid = torch.meshgrid(xx, yy, indexing='xy')
grid_xy = torch.stack([xx_grid.ravel(), yy_grid.ravel()], dim=1).to(device)

# Render original frame
print("Rendering original frame...")
px, py = compute_points(actual_t, device=device)
points_norm = torch.stack([px, py], dim=1)
intensities = compute_intensities(points_norm, grid_xy, device=device)
original_frame = intensities.cpu().numpy().reshape(GRID_RES, GRID_RES)

print(f"Original frame stats: min={original_frame.min():.4f}, max={original_frame.max():.4f}, mean={original_frame.mean():.4f}")
print("-" * 50)

# Test with increasing blur strength
blur_strengths = np.arange(0.5, 10.0, 0.5)  # sigma values from 0.5 to 10 (skip perfect match)
psnr_values = []

print("Applying Gaussian blur and calculating PSNR...")
for sigma in blur_strengths:
    # Apply Gaussian blur
    blurred_frame = gaussian_filter(original_frame, sigma=sigma)
    
    # Calculate PSNR against original (data_range=1.0 since intensities are in [0, 1])
    psnr_val = psnr(original_frame, blurred_frame, data_range=1.0)
    psnr_values.append(psnr_val)
    print(f"Sigma: {sigma:.1f} -> PSNR: {psnr_val:.2f} dB")

print("-" * 50)

# Save sample images at different blur levels
print("Saving sample images...")
# Show original + blurred versions
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Original vs Blurred (t=0.5)', fontsize=16)

# First image is the original
axes[0].imshow(original_frame, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Original\n(σ=0.0)')
axes[0].axis('off')

# Then show progressively more blurred versions
sample_sigmas = [0.5, 2.5, 5.0, 9.5]
for i, sigma in enumerate(sample_sigmas, 1):
    blurred_sample = gaussian_filter(original_frame, sigma=sigma)
    # Find corresponding PSNR value
    sigma_idx = np.where(blur_strengths == sigma)[0][0]
    psnr_val = psnr_values[sigma_idx]
    
    axes[i].imshow(blurred_sample, cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f'σ={sigma:.1f}\nPSNR={psnr_val:.2f} dB')
    axes[i].axis('off')

plt.tight_layout()
sample_path = '.cache/frames/blur_samples.png'
plt.savefig(sample_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sample images saved to {sample_path}")

# Plot PSNR vs blur strength
print("Plotting PSNR curve...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(blur_strengths, psnr_values, 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Gaussian Blur Strength (σ)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('PSNR vs Gaussian Blur Strength (400x400 frame at t=0.5)', fontsize=14)
ax.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines
ax.set_xlim(0, max(blur_strengths))
# Use log scale for better visualization of PSNR degradation
ax.set_yscale('log')
# Handle ylim carefully for finite values
valid_psnrs = [p for p in psnr_values if np.isfinite(p) and p > 0]
ax.set_ylim(min(valid_psnrs) * 0.95, max(valid_psnrs) * 1.05)

# Add some annotations
for i in range(1, len(blur_strengths), 4):  # Annotate every 4th point
    ax.plot(blur_strengths[i], psnr_values[i], 'ro', markersize=8)
    ax.annotate(f'{psnr_values[i]:.1f} dB', 
                (blur_strengths[i], psnr_values[i]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=9, color='red')

plt.tight_layout()
plot_path = '.cache/frames/blur_psnr_curve.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"PSNR curve saved to {plot_path}")

# Print summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"PSNR at sigma=0.5: {psnr_values[0]:.2f} dB")
idx_2_5 = np.where(blur_strengths == 2.5)[0][0]
idx_5_0 = np.where(blur_strengths == 5.0)[0][0]
idx_9_5 = len(psnr_values) - 1
print(f"PSNR at sigma=2.5: {psnr_values[idx_2_5]:.2f} dB")
print(f"PSNR at sigma=5.0: {psnr_values[idx_5_0]:.2f} dB")
print(f"PSNR at sigma=9.5: {psnr_values[idx_9_5]:.2f} dB")
print(f"PSNR decrease: {psnr_values[0] - psnr_values[-1]:.2f} dB")
print("=" * 50)
print("\nTest complete!")

