"""
Neural network model for shader approximation with WINNER initialization.
"""
import torch
import torch.nn as nn
import math
from collections import OrderedDict  # For forward_with_activations if needed later
import torch.fft as fft  # For spectral centroid


class SineLayer(nn.Module):
    """SIREN sine layer, per official implementation."""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
                nn.init.uniform_(self.linear.weight, -bound, bound)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                nn.init.uniform_(self.linear.weight, -bound, bound)
            # Biases default to zero (PyTorch nn.Linear behavior); no explicit init

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ShaderMLP(nn.Module):
    """SIREN MLP for shader intensity approximation, with WINNER init.
    
    Input: (x, y, t) normalized [-1, 1]
    Output: intensity in [0, 1]
    """
    def __init__(self, input_dim=3, hidden_dim=1024, num_sine_layers=3, 
                 first_omega_0=180.0, hidden_omega_0=60.0, spectral_centroid=None):
        super().__init__()
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        # WINNER: Compute noise scales if spectral_centroid provided
        self.s0 = 0.0
        self.s1 = 0.0
        if spectral_centroid is not None:
            C = 1.0
            s_max0, a, b = 50.0, 5.0, 0.4
            psi_norm = spectral_centroid / C
            self.s0 = s_max0 * (1 - math.exp(-a * psi_norm))
            self.s1 = b * psi_norm
            print(f"WINNER: ψ={spectral_centroid:.2f}, s0={self.s0:.2f}, s1={self.s1:.2f}")
        
        net = []
        self.first_layer = SineLayer(input_dim, hidden_dim, is_first=True, omega_0=first_omega_0)
        net.append(self.first_layer)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(1, num_sine_layers):
            layer = SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=hidden_omega_0)
            self.hidden_layers.append(layer)
            net.append(layer)
        
        net.append(nn.Linear(hidden_dim, 1))
        with torch.no_grad():
            fan_in = hidden_dim
            bound = math.sqrt(6 / fan_in) / hidden_omega_0
            nn.init.uniform_(net[-1].weight, -bound, bound)

        
        self.net = nn.Sequential(*net)
        
        # Apply WINNER noise post-init (to first two layers)
        self._apply_winner_noise()

    def forward(self, xy_t):
        # Detach/require_grad for potential autograd ops (e.g., gradients/laplacian)
        x = xy_t.clone().detach().requires_grad_(True)
        logits = self.net(x)
        return torch.sigmoid(logits)

    def forward_logits(self, xy_t):
        """Return raw logits (no sigmoid). Useful for BCEWithLogitsLoss during training."""
        x = xy_t.clone().detach().requires_grad_(True)
        return self.net(x)

    def predict(self, xy_t):
        """Inference helper used by animation/rendering code.
        Runs the network in eval mode with no grad and returns the output tensor.
        """
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                return self.forward(xy_t)
        finally:
            if was_training:
                self.train()

    def forward_with_activations(self, coords, retain_grad=False):
        """Optional: For visualizing activations (as in Colab)."""
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
                activations[f'SineLayer_{activation_count}'] = x
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
                activations[f'Linear_{activation_count}'] = x
                activation_count += 1
        return activations

    def _apply_winner_noise(self):
        """Add Gaussian noise to first two layers per WINNER."""
        with torch.no_grad():
            if self.s0 > 0:
                eta0 = torch.normal(0, self.s0 / self.first_omega_0, size=self.first_layer.linear.weight.shape)
                self.first_layer.linear.weight.add_(eta0)
            if len(self.hidden_layers) > 0 and self.s1 > 0:
                eta1 = torch.normal(0, self.s1 / self.hidden_omega_0, size=self.hidden_layers[0].linear.weight.shape)
                self.hidden_layers[0].linear.weight.add_(eta1)


def compute_spectral_centroid(frame: torch.Tensor) -> float:
    """Compute 2D spectral centroid ψ for image frame (WINNER init).
    
    Args:
        frame: (H, W) grayscale tensor [0,1]
    Returns:
        ψ: Spectral centroid (scalar)
    """
    device = frame.device
    dtype = frame.dtype
    fft_frame = fft.fft2(frame)
    mag = torch.abs(fft_frame) ** 2
    H, W = frame.shape
    kx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)
    ky = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
    k = torch.sqrt(kx**2 + ky**2)
    numerator = 2 * torch.sum(k * mag)
    denominator = torch.sum(mag)
    psi = numerator / denominator if denominator > 0 else 0.0
    return float(psi)

 