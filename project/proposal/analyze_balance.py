import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
print("Loading dataset...")
data = torch.load('.cache/ds/shader_dataset.pt', map_location='cpu')
x, y, t, intensity = data['x'], data['y'], data['t'], data['intensity']

# Convert to numpy for analysis and plotting
intensity_np = intensity.numpy()
x_np = x.numpy()
y_np = y.numpy()
t_np = t.numpy()

print(f"Total samples: {len(intensity_np):,}")
print(f"Intensity range: [{np.min(intensity_np):.4f}, {np.max(intensity_np):.4f}]")
print(f"Intensity mean: {np.mean(intensity_np):.4f}")
print(f"Intensity std: {np.std(intensity_np):.4f}")

# Class balance analysis (treating >0.5 as positive)
binary_intensity = (intensity_np > 0.5).astype(int)
positive_ratio = np.mean(binary_intensity)
negative_ratio = 1 - positive_ratio

print("\n=== Binary Class Balance ===")
print(f"Positive (>0.5): {np.sum(binary_intensity):,} ({positive_ratio:.2%})")
print(f"Negative (<=0.5): {np.sum(1-binary_intensity):,} ({negative_ratio:.2%})")
print(f"Class ratio: {positive_ratio:.2%} / {negative_ratio:.2%}")
if positive_ratio > 0.4 and positive_ratio < 0.6:
    print("✓ Classes are reasonably balanced")
else:
    print("✗ Classes are imbalanced!")

# Create histogram and KDE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with bins
ax1 = axes[0, 0]
n_bins = 50
ax1.hist(intensity_np, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Intensity')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Intensity Distribution (Histogram, {n_bins} bins)')
ax1.grid(True, alpha=0.3)
ax1.axvline(0.5, color='red', linestyle='--', label='Binary threshold (0.5)')
ax1.legend()

# 2. KDE (Kernel Density Estimate)
ax2 = axes[0, 1]
kde = stats.gaussian_kde(intensity_np)
x_kde = np.linspace(intensity_np.min(), intensity_np.max(), 1000)
density = kde(x_kde)
ax2.plot(x_kde, density, 'b-', linewidth=2, label='KDE')
ax2.fill_between(x_kde, density, alpha=0.3)
ax2.set_xlabel('Intensity')
ax2.set_ylabel('Density')
ax2.set_title('Intensity Distribution (KDE)')
ax2.grid(True, alpha=0.3)
ax2.axvline(0.5, color='red', linestyle='--', label='Binary threshold (0.5)')
ax2.legend()

# 3. Binary class balance
ax3 = axes[1, 0]
classes = ['Negative\n(<=0.5)', 'Positive\n(>0.5)']
counts = [np.sum(1-binary_intensity), np.sum(binary_intensity)]
colors = ['lightcoral', 'lightblue']
bars = ax3.bar(classes, counts, color=colors, edgecolor='black')
ax3.set_ylabel('Count')
ax3.set_title('Binary Class Distribution')
ax3.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({count/len(intensity_np):.1%})',
             ha='center', va='bottom', fontsize=10)

# 4. Intensity by time
ax4 = axes[1, 1]
# Sample subset for visualization
sample_idx = np.random.choice(len(intensity_np), min(50000, len(intensity_np)), replace=False)
ax4.scatter(t_np[sample_idx], intensity_np[sample_idx], alpha=0.1, s=0.5, color='purple')
ax4.set_xlabel('Time (normalized)')
ax4.set_ylabel('Intensity')
ax4.set_title('Intensity over Time (50k sample)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '.cache/frames/class_balance_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_path}")
plt.show()

# Additional statistics
print("\n=== Intensity Statistics ===")
print(f"Zero values: {np.sum(intensity_np == 0):,} ({np.mean(intensity_np == 0):.2%})")
print(f"One values: {np.sum(intensity_np == 1):,} ({np.mean(intensity_np == 1):.2%})")
print(f"Continuous values (0 < x < 1): {np.sum((intensity_np > 0) & (intensity_np < 1)):,} ({np.mean((intensity_np > 0) & (intensity_np < 1)):.2%})")

# Percentiles
print("\n=== Intensity Percentiles ===")
percentiles = [0, 10, 25, 50, 75, 90, 100]
for p in percentiles:
    val = np.percentile(intensity_np, p)
    print(f"{p:3d}th percentile: {val:.4f}")

