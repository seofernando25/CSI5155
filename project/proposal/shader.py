"""
PyTorch-accelerated shader computation functions for GPU.
"""
import torch

# Simulation constants
W = 400.0  # Canvas width
NORM_SCALE = (W / 2.0)
NUM_I = 10000  # Number of particles
T_MAX = 12.5  # Time period (periodic boundary)

# Point rendering constants
POINT_RADIUS = 1.0


def compute_points(t, num_i=NUM_I, device='cuda'):
    """
    Compute particle positions at time t, normalized to [-1, 1] centered at origin.
    Uses PyTorch for GPU acceleration.
    
    Args:
        t: Time value
        num_i: Number of particles
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        px, py: Particle positions normalized to [-1, 1], centered at (0, 0)
    """
    i_arr = torch.flip(torch.arange(num_i, dtype=torch.float32, device=device), [0])
    y = i_arr / 790.0
    int_y = torch.floor(y).long()
    k_base = torch.where(y < 5, 6 + torch.sin(int_y ^ 1).float() * 6, 4 + torch.cos(y))
    k = k_base * torch.cos(i_arr + t / 4)
    e = y / 3 - 13
    d = torch.sqrt(k**2 + e**2) + torch.sin(e / 4 - t) / 3
    q = y * k / 5 * (2 + torch.sin(d * 2 + y - t * 4))
    c = d / 3 - t / 2 + (i_arr % 2).float()
    
    # Canvas coordinates [0, 400]
    px_canvas = q + 90 * torch.cos(c) + 200
    py_canvas = q * torch.sin(c) + d * 29 - 170
    
    # Normalize to [-1, 1] centered at origin
    px = (px_canvas - NORM_SCALE) / NORM_SCALE
    py = -(py_canvas - NORM_SCALE) / NORM_SCALE
    return px, py


def compute_intensities(points_norm, grid_xy_norm, device='cuda'):
    """
    Compute intensities using PyTorch distance computation on GPU.
    This should be faster than scipy.spatial.KDTree, especially on GPU.
    
    Args:
        points_norm: (N, 2) tensor of particle positions
        grid_xy_norm: (M, 2) tensor of grid positions
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        intensities: (M,) tensor of intensities [0, 1]
    """
    if isinstance(points_norm, torch.Tensor) and points_norm.device.type != device:
        points_norm = points_norm.to(device)
    if isinstance(grid_xy_norm, torch.Tensor) and grid_xy_norm.device.type != device:
        grid_xy_norm = grid_xy_norm.to(device)
    
    # Compute distances: (M, N) distance matrix
    # Use broadcasting: grid_xy (M, 1, 2) - points_norm (1, N, 2)
    diff = grid_xy_norm.unsqueeze(1) - points_norm.unsqueeze(0)  # (M, N, 2)
    distances = torch.norm(diff, dim=2)  # (M, N)
    
    # Find minimum distance to any point
    min_dists = torch.min(distances, dim=1)[0]  # (M,)
    
    # Check if within radius
    r_norm = POINT_RADIUS / NORM_SCALE
    intensities = (min_dists <= r_norm).float()
    
    return intensities


def compute_global_bbox(num_samples=500, expand=5.0, device='cuda'):
    """
    Compute the global bounding box of the shader animation over time.
    
    Args:
        num_samples: Number of time samples to use for bbox computation
        expand: Expansion factor for the bbox (in pixel units)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        min_x, max_x, min_y, max_y: Bounding box in normalized space [-1, 1]
    """
    import numpy as np
    
    t_bbox_samples = np.linspace(0, T_MAX, num_samples)
    all_px_norm, all_py_norm = [], []
    for t in t_bbox_samples:
        px_norm, py_norm = compute_points(t, device=device)
        all_px_norm.extend(px_norm.cpu().numpy())
        all_py_norm.extend(py_norm.cpu().numpy())
    
    all_px_norm = np.array(all_px_norm)
    all_py_norm = np.array(all_py_norm)
    
    min_x, max_x = np.min(all_px_norm), np.max(all_px_norm)
    min_y, max_y = np.min(all_py_norm), np.max(all_py_norm)
    
    # Expand bbox in normalized space
    expand_norm = expand / NORM_SCALE
    min_x = max(-1, min_x - expand_norm)
    max_x = min(1, max_x + expand_norm)
    min_y = max(-1, min_y - expand_norm)
    max_y = min(1, max_y + expand_norm)
    
    return min_x, max_x, min_y, max_y


# Compile the functions for better performance
if hasattr(torch, 'compile'):
    compute_points = torch.compile(compute_points, mode='reduce-overhead', fullgraph=True)
    compute_intensities = torch.compile(compute_intensities, mode='reduce-overhead', fullgraph=True)

