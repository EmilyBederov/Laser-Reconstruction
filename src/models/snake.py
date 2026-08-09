"""Snake activation: x + (1/alpha) * sin^2(alpha*x).

From "Neural Networks Fail to Learn Periodic Functions and How to Fix It"
(Ziyin et al., NeurIPS 2020) and adopted by BigVGAN for waveform synthesis.
The periodic bias preserves harmonic structure that ReLU/GELU smooth out —
particularly useful for voiced high-frequency content in vocoders.
"""
import torch
import torch.nn as nn


class Snake(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        # per-channel learnable frequency parameter, broadcast over time
        self.alpha = nn.Parameter(torch.full((1, channels, 1), float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        a = self.alpha + 1e-9
        return x + (1.0 / a) * torch.sin(a * x).pow(2)
