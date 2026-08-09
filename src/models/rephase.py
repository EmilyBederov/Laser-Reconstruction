"""
MP-SENet-style re-phasing head.

The base model (CCF + Denoiser + Reconstructor) already produces a GOOD magnitude
(STOI ~0.78) but Griffin-Lim-quality phase. This module KEEPS that magnitude and
learns only a better phase, then recombines via iSTFT:

    y_hat (frozen base) ──STFT──► magnitude |S|        (kept, detached)
                                  phase_orig ∠S        (the bad implicit phase)
    PhaseDecoder([log|S|, cos∠S, sin∠S]) ──► phase_new
    y_out = iSTFT( |S| · exp(j·phase_new) )

The phase decoder is MP-SENet style: a dilated dense block + two parallel conv
branches (pseudo-real / pseudo-imaginary) → phase = atan2(I, R), which is naturally
wrapped to (−π, π].  Train with anti-wrapping phase losses (see losses.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedDenseBlock(nn.Module):
    """MP-SENet dense block: n conv layers, dilation 2^i, dense (concat) connections."""
    def __init__(self, ch: int, n: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * (i + 1), ch, kernel_size=3,
                          padding=2 ** i, dilation=2 ** i),
                nn.InstanceNorm2d(ch, affine=True),
                nn.PReLU(ch),
            ) for i in range(n)
        ])

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return feats[-1]


class PhaseDecoder(nn.Module):
    """(B, in_ch, F, T) → wrapped phase (B, F, T) via parallel R/I convs + atan2."""
    def __init__(self, in_ch: int = 3, hidden: int = 48, n_dense: int = 4):
        super().__init__()
        self.inp   = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.InstanceNorm2d(hidden, affine=True), nn.PReLU(hidden))
        self.dense = DilatedDenseBlock(hidden, n=n_dense)
        self.conv_r = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)   # pseudo-real
        self.conv_i = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)   # pseudo-imag

    def forward(self, x):                       # x: (B, in_ch, F, T)
        h = self.dense(self.inp(x))
        r = self.conv_r(h)                      # (B, 1, F, T)
        i = self.conv_i(h)
        return torch.atan2(i, r).squeeze(1)     # (B, F, T) in (−π, π]


class RephaseHead(nn.Module):
    """Keep the base model's magnitude, predict a new phase, recombine via iSTFT."""
    def __init__(self, n_fft: int = 1024, hop: int = 128, hidden: int = 48):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        # input channels: log-magnitude + cos/sin of the original (bad) phase
        self.decoder = PhaseDecoder(in_ch=3, hidden=hidden)

    def _stft(self, wav):                        # wav: (B, T_wav) → complex (B, F, T)
        return torch.stft(wav, self.n_fft, self.hop, self.n_fft,
                          self.win.to(wav.device), center=True, return_complex=True)

    def forward(self, y_hat):                    # y_hat: (B, 1, T_wav) from FROZEN base
        L = y_hat.shape[-1]
        S = self._stft(y_hat[:, 0].float())      # (B, F, T) complex
        mag   = S.abs().detach()                 # GOOD magnitude — kept, frozen
        ph0   = torch.angle(S).detach()          # original (bad) phase
        feat  = torch.stack([torch.log(mag + 1e-5),
                             torch.cos(ph0), torch.sin(ph0)], dim=1)   # (B, 3, F, T)
        phase = self.decoder(feat)               # (B, F, T) new phase
        spec  = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        y_out = torch.istft(spec, self.n_fft, self.hop, self.n_fft,
                            self.win.to(y_hat.device), center=True, length=L)
        return y_out.unsqueeze(1), phase         # (B, 1, T_wav), (B, F, T)
