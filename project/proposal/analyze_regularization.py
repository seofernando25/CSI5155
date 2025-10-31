import os
import argparse
import math
import torch
import numpy as np
from typing import Dict, List

from model import ShaderMLP
from shader import W, T_MAX


def load_model(model_path: str, device: torch.device) -> ShaderMLP:
    model = ShaderMLP().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Handle compiled checkpoints (_orig_mod prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        abs_t = t.abs()
        return {
            'numel': int(t.numel()),
            'l1': float(abs_t.sum().item()),
            'l2': float(torch.sqrt((t * t).sum()).item()),
            'mean_abs': float(abs_t.mean().item()),
            'max_abs': float(abs_t.max().item()),
        }


def layer_sparsity(t: torch.Tensor, thresholds: List[float]) -> Dict[str, float]:
    with torch.no_grad():
        out = {}
        abs_t = t.abs()
        numel = t.numel()
        for th in thresholds:
            frac = float((abs_t < th).sum().item()) / max(1, numel)
            out[f'frac_below_{th}'] = frac
        return out


def neuron_connectivity(weight: torch.Tensor, row_threshold: float, col_threshold: float) -> Dict[str, int]:
    # weight shape: [out_features, in_features]
    with torch.no_grad():
        row_l1 = weight.abs().sum(dim=1)
        col_l1 = weight.abs().sum(dim=0)
        dead_rows = int((row_l1 < row_threshold).sum().item())
        dead_cols = int((col_l1 < col_threshold).sum().item())
        return {
            'near_zero_output_neurons': dead_rows,
            'near_zero_input_channels': dead_cols,
        }


def build_full_grid(device: torch.device) -> torch.Tensor:
    grid_res = int(W)
    x_lin = torch.linspace(-1.0, 1.0, grid_res, device=device)
    y_lin = torch.linspace(-1.0, 1.0, grid_res, device=device)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing='xy')
    grid_xy_full = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    inputs_full = torch.empty((grid_xy_full.shape[0], 3), device=device, dtype=torch.float32)
    inputs_full[:, :2] = grid_xy_full
    return inputs_full


def activation_variance(model: ShaderMLP, device: torch.device, time_samples: List[float]) -> Dict[str, Dict[str, float]]:
    inputs_full = build_full_grid(device)
    t_col = inputs_full[:, 2:3]
    # Collect activation variances per recorded layer
    variances: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for t_norm in time_samples:
            t_col.fill_(float(t_norm))
            acts = model.forward_with_activations(inputs_full, retain_grad=False)
            for name, a in acts.items():
                # Skip the raw input
                if name == 'input':
                    continue
                v = torch.var(a, dim=0).mean().item() if a.ndim == 2 else torch.var(a).item()
                key = f'{name}'
                if key not in variances:
                    variances[key] = {'mean_var': 0.0, 'count': 0}
                variances[key]['mean_var'] += v
                variances[key]['count'] += 1
    # Average across time samples
    for k in list(variances.keys()):
        variances[k]['mean_var'] /= max(1, variances[k]['count'])
    return variances


def analyze(model_path: str, device_str: str, report_path: str) -> None:
    device = torch.device(device_str if device_str in ['cpu', 'cuda'] else 'cpu')
    model = load_model(model_path, device)

    lines: List[str] = []
    lines.append(f'Model path: {model_path}')
    lines.append(f'Device: {device}')
    lines.append('')

    thresholds = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
    row_th = 1e-6
    col_th = 1e-6

    lines.append('Per-layer weight statistics:')
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            b = module.bias.data if module.bias is not None else None
            st = tensor_stats(w)
            sp = layer_sparsity(w, thresholds)
            conn = neuron_connectivity(w, row_th, col_th)
            lines.append(f'- Linear[{idx}] {name}: weight shape={tuple(w.shape)}, params={st["numel"]}')
            lines.append(f'  L1={st["l1"]:.4e} L2={st["l2"]:.4e} mean|w|={st["mean_abs"]:.4e} max|w|={st["max_abs"]:.4e}')
            lines.append('  Sparsity (fraction |w| < th): ' + ', '.join([f'{k}={v:.3f}' for k, v in sp.items()]))
            lines.append(f'  Near-zero neurons (row<{row_th}, col<{col_th}): rows={conn["near_zero_output_neurons"]}, cols={conn["near_zero_input_channels"]}')
            if b is not None:
                bst = tensor_stats(b)
                lines.append(f'  Bias: params={bst["numel"]} mean|b|={bst["mean_abs"]:.4e} max|b|={bst["max_abs"]:.4e}')
            idx += 1
    lines.append('')

    # Activation variance across a few times; near-zero variance suggests constant/unused channels
    lines.append('Activation variance (averaged across time samples):')
    t_samples = [0.1, 0.5, 0.9]
    av = activation_variance(model, device, t_samples)
    for lname, stats in av.items():
        lines.append(f'- {lname}: mean variance={stats["mean_var"]:.4e}')
    lines.append('')

    # Write report
    os.makedirs('.cache/frames', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Regularization analysis written to: {report_path}")
    # Also print a brief summary to stdout
    print('\n'.join(lines[:min(len(lines), 40)]))


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze model regularization and potential unused weights')
    parser.add_argument('--model_path', type=str, default='.cache/models/best_model.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--report_path', type=str, default='.cache/frames/regularization_report.txt')
    args = parser.parse_args()
    analyze(args.model_path, args.device, args.report_path)


if __name__ == '__main__':
    main()



