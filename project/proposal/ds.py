import torch
import time
import os
from tqdm import tqdm
from shader import compute_points, W, NUM_I, T_MAX, compute_global_bbox
from shader import compute_intensities

torch.manual_seed(42)  # For reproducibility

# Setup output directory
os.makedirs('.cache/ds', exist_ok=True)

# Constants
GRID_RES = int(W)
SUBSAMPLE_FRAC = 0.15
NUM_T = int(T_MAX * 60)
STRATIFIED_SAMPLING = True  # If True, balance ones and zeros. If False, use random subsample.

USE_BBOX = False  # Sample across the full normalized area [-1, 1]^2
EXPAND = 5.0  # Increased for better tendril/glow coverage
NUM_BBOX_SAMPLES = 100  # More for accurate global span

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def subsample_indices_torch(n_total, n_keep):
    """PyTorch version of subsample_indices"""
    if n_keep >= n_total:
        return torch.arange(n_total)
    idxs = torch.randperm(n_total)[:n_keep]
    return idxs

def process_coverage_frame(args):
    idx, t, points_norm, grid_xy_norm, subsample_frac, t_max, stratified = args
    
    # Compute intensities (will use GPU if available in the shader function)
    intensities = compute_intensities(points_norm, grid_xy_norm, device='cpu')
    
    if stratified:
        # Stratified sampling: balance ones and zeros
        ones_mask = intensities > 0.5
        zeros_mask = intensities <= 0.5
        
        ones_idx = torch.where(ones_mask)[0]
        zeros_idx = torch.where(zeros_mask)[0]
        
        n_ones = len(ones_idx)
        n_zeros = len(zeros_idx)
        n_target = min(n_ones, n_zeros, int(len(grid_xy_norm) * subsample_frac))
        
        # Keep all ones if possible, otherwise subsample
        if n_ones <= n_target:
            ones_selected = ones_idx
        else:
            ones_selected = ones_idx[torch.randperm(n_ones)[:n_target]]
        
        # Sample zeros to match
        if n_zeros <= n_target:
            zeros_selected = zeros_idx
        else:
            zeros_selected = zeros_idx[torch.randperm(n_zeros)[:n_target]]
        
        idxs = torch.cat([ones_selected, zeros_selected])
    else:
        # Original random sampling
        n_keep = int(len(grid_xy_norm) * subsample_frac)
        idxs = subsample_indices_torch(len(grid_xy_norm), n_keep)
    
    grid_xy_s = grid_xy_norm[idxs]
    intensities_s = intensities[idxs]
    
    return {
        'x': grid_xy_s[:, 0].cpu().numpy(),
        'y': grid_xy_s[:, 1].cpu().numpy(),
        't': torch.full((len(grid_xy_s),), t / t_max).cpu().numpy(),
        'intensity': intensities_s.cpu().numpy()
    }

# Global bbox computation in normalized space [-1, 1]
print("Computing global bounding box...")
min_x_norm, max_x_norm, min_y_norm, max_y_norm = compute_global_bbox(num_samples=NUM_BBOX_SAMPLES, expand=EXPAND)

bbox_area = (max_x_norm - min_x_norm) * (max_y_norm - min_y_norm)
full_area = 4.0  # Full normalized area is [-1,1] x [-1,1] = 4
print(f"Global bbox (normalized): [{min_x_norm:.3f}, {max_x_norm:.3f}] x [{min_y_norm:.3f}, {max_y_norm:.3f}], area={bbox_area:.3f} ({bbox_area/full_area:.1%} of canvas)")

t_values = torch.linspace(0, T_MAX, NUM_T)
all_x, all_y, all_t, all_intensity = [], [], [], []

# Pre-compute points in normalized space
print("Generating all points...")
all_points_norm = torch.zeros((NUM_T, NUM_I, 2), device=device)
compute_start = time.time()
for idx, t in enumerate(t_values):
    if idx % 100 == 0:
        print(f"Computing points: {idx}/{NUM_T}")
    px_norm, py_norm = compute_points(t.item(), device=device)
    all_points_norm[idx, :, 0] = px_norm
    all_points_norm[idx, :, 1] = py_norm
compute_time = time.time() - compute_start
print(f"Point generation completed in {compute_time:.1f}s")

print("Preparing grid...")

# Grid in normalized space: Bbox or full [-1, 1]
if USE_BBOX:
    # Compute grid resolution based on bbox size
    bbox_width = max_x_norm - min_x_norm
    bbox_height = max_y_norm - min_y_norm
    x_grid_len = int(GRID_RES * bbox_width / 2.0)
    y_grid_len = int(GRID_RES * bbox_height / 2.0)
    x_grid = torch.linspace(min_x_norm, max_x_norm, x_grid_len)
    y_grid = torch.linspace(min_y_norm, max_y_norm, y_grid_len)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')
    grid_xy_norm = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    print(f"Bbox grid size: {len(grid_xy_norm)} queries/frame")
    print(f"  Grid dimensions: {x_grid_len} x {y_grid_len}")
else:
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, GRID_RES), torch.linspace(-1, 1, GRID_RES), indexing='xy')
    grid_xy_norm = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    print(f"Full grid size: {len(grid_xy_norm)} queries/frame")

# Compute expected sampled points per frame
initial_grid_size = len(grid_xy_norm)
if STRATIFIED_SAMPLING:
    # With stratified sampling, we sample up to 2 * min(n_ones, n_zeros, grid_size * subsample_frac)
    # In the worst case (balanced ones/zeros), this is approximately 2 * (grid_size * subsample_frac)
    max_sampled_stratified = 2 * int(initial_grid_size * SUBSAMPLE_FRAC)
    print("\nSampling strategy: STRATIFIED")
    print(f"  Initial grid points per frame: {initial_grid_size:,}")
    print(f"  Subsample fraction: {SUBSAMPLE_FRAC} ({SUBSAMPLE_FRAC*100:.1f}%)")
    print(f"  Target per class (ones/zeros): up to {int(initial_grid_size * SUBSAMPLE_FRAC):,}")
    print(f"  Maximum sampled points per frame (if balanced): up to ~{max_sampled_stratified:,}")
    print("  Note: Actual count varies per frame based on ones/zeros distribution")
else:
    sampled_per_frame = int(initial_grid_size * SUBSAMPLE_FRAC)
    print("\nSampling strategy: RANDOM")
    print(f"  Initial grid points per frame: {initial_grid_size:,}")
    print(f"  Subsample fraction: {SUBSAMPLE_FRAC} ({SUBSAMPLE_FRAC*100:.1f}%)")
    print(f"  Sampled points per frame: {sampled_per_frame:,}")

print("Processing coverage and subsampling...")
process_start = time.time()

# Process each frame sequentially in main process (avoid CUDA fork issues)
frame_results = []
sampled_counts = []
for idx in tqdm(range(NUM_T), desc="Processing frames"):
    args = (idx, t_values[idx].item(), all_points_norm[idx].detach(), grid_xy_norm.detach(), 
            SUBSAMPLE_FRAC, T_MAX, STRATIFIED_SAMPLING)
    result = process_coverage_frame(args)
    frame_results.append(result)
    sampled_counts.append(len(result['x']))

# Print statistics about sampled points per frame
if sampled_counts:
    sampled_counts = torch.tensor(sampled_counts)
    print("\nActual sampled points per frame statistics:")
    print(f"  Mean: {sampled_counts.float().mean():.1f}")
    print(f"  Median: {sampled_counts.median().item():.1f}")
    print(f"  Min: {sampled_counts.min().item():,}")
    print(f"  Max: {sampled_counts.max().item():,}")
    print(f"  Std: {sampled_counts.float().std():.1f}")

print("Converting results to tensors...")

# Flatten and convert to tensors
all_x = torch.cat([torch.tensor(result['x']) for result in frame_results])
all_y = torch.cat([torch.tensor(result['y']) for result in frame_results])
all_t = torch.cat([torch.tensor(result['t']) for result in frame_results])
all_intensity = torch.cat([torch.tensor(result['intensity']) for result in frame_results])

print("Saving dataset...")
# Save as PyTorch tensors
output_path = '.cache/ds/shader_dataset.pt'
torch.save({
    'x': all_x,
    'y': all_y,
    't': all_t,
    'intensity': all_intensity
}, output_path)
process_time = time.time() - process_start
total_time = compute_time + process_time
print(f"Dataset saved to {output_path}: {len(all_x)} samples")
print(f"Total time: {total_time:.1f}s (points: {compute_time:.1f}s, processing: {process_time:.1f}s)")
print(f"Non-zero fraction: {torch.mean((all_intensity > 0.1).float()).item():.4f}")
print(f"Intensity: min={all_intensity.min().item():.4f}, max={all_intensity.max().item():.4f}, mean={all_intensity.mean().item():.4f}")
