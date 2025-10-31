import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from shader import compute_intensities, compute_points, W, T_MAX


def render_shader_frame(grid_res: int, t_norm: float) -> np.ndarray:
    device = 'cpu'
    # Normalized full grid in [-1, 1]
    x_lin = torch.linspace(-1.0, 1.0, grid_res)
    y_lin = torch.linspace(-1.0, 1.0, grid_res)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing='xy')
    grid_xy_full = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    # Shader points at physical time (t_norm in [0,1])
    t_phys = float(t_norm * T_MAX)
    px, py = compute_points(t_phys, device=device)
    points = torch.stack([px, py], dim=1)

    with torch.no_grad():
        gt_flat = compute_intensities(points, grid_xy_full, device=device)
        frame = gt_flat.reshape(grid_res, grid_res).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return frame


def main():
    grid_res = int(W)
    # Choose a mid-cycle time
    t_norm = 0.5
    shader_frame = render_shader_frame(grid_res, t_norm)
    black = np.zeros_like(shader_frame, dtype=np.float32)

    score = ssim(black, shader_frame, data_range=1.0)
    print(f"SSIM(black vs shader @W={grid_res}, t_norm={t_norm}): {float(score):.6f}")


if __name__ == "__main__":
    main()



