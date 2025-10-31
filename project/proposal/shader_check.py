import numpy as np
import torch
import time
from shader import compute_points, compute_intensities, compute_global_bbox

print("=" * 60)
print("Shader Functionality Test")
print("=" * 60)

# Test parameters
test_times = [0.0, 3.125, 6.25, 9.375, 12.5]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

all_passed = True

for t in test_times:
    print(f"Testing time t = {t:.3f}")
    
    try:
        # Compute points
        px, py = compute_points(t, device=device)
        points = torch.stack([px, py], dim=1)
        
        # Test intensities on a small grid
        W_test = 100  # Smaller grid for faster testing
        xx, yy = np.meshgrid(np.linspace(-1, 1, W_test), np.linspace(-1, 1, W_test))
        grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        grid_xy_torch = torch.tensor(grid_xy, dtype=torch.float32).to(device)
        
        intensities = compute_intensities(points, grid_xy_torch, device=device)
        
        print(f"  ✓ Computed {len(points)} points")
        print(f"  ✓ Computed {len(intensities)} intensities")
        print(f"  ✓ Intensity range: [{intensities.min():.4f}, {intensities.max():.4f}], mean: {intensities.mean():.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print()

print("=" * 60)
if all_passed:
    print("✓ ALL TESTS PASSED!")
else:
    print("✗ SOME TESTS FAILED")
print("=" * 60)

# Performance benchmark
print("\n" + "=" * 60)
print("Performance Benchmark")
print("=" * 60)

W_bench = 400  # 400x400 grid
xx, yy = np.meshgrid(np.linspace(-1, 1, W_bench), np.linspace(-1, 1, W_bench))
grid_xy_bench = np.stack([xx.ravel(), yy.ravel()], axis=1)

# Test multiple time points
test_times = [0.0, 3.125, 6.25, 9.375, 12.5]
num_iterations = 10

print(f"Benchmarking {W_bench}x{W_bench} grid across {len(test_times)} time samples")
print(f"Running {num_iterations} iterations...")

print(f"\nPyTorch Shader ({device}) implementation:")
grid_xy_torch_bench = torch.tensor(grid_xy_bench, dtype=torch.float32).to(device)
all_times = []
for t_idx, test_t in enumerate(test_times):
    times = []
    for i in range(num_iterations + 1):  # +1 for warmup
        start = time.time()
        px, py = compute_points(test_t, device=device)
        points_norm = torch.stack([px, py], dim=1)
        intensities = compute_intensities(points_norm, grid_xy_torch_bench, device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        if i > 0:  # Skip warmup
            times.append(elapsed)
    all_times.extend(times)

avg_time = np.mean(all_times)
std_time = np.std(all_times)
print(f"  Average: {avg_time:.1f} ± {std_time:.1f} ms/frame ({1000/avg_time:.1f} FPS)")
print("=" * 60)

