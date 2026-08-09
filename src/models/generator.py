import torch
import torch.nn as nn
from .denoiser import DenoiseClass
from .bwe_reconstruction import ReconstrucClass


class CrossChannelFusion(nn.Module):
    """Symmetric bidirectional fusion: each channel enhances the other.
    Same gate_net is called twice with reversed channel order (weight sharing).
    alpha=0 init: no contribution on first step, grows as training progresses."""
    def __init__(self, n_mel: int = 80):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv1d(n_mel * 2, n_mel, kernel_size=7, padding=3),
            nn.Sigmoid())
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        # x_spec: (B, 2, F, T)
        B, _, F, T = x_spec.shape
        ch_x = x_spec[:, 0]  # (B, F, T)
        ch_y = x_spec[:, 1]
        gate_xy = self.gate_net(torch.cat([ch_x, ch_y], dim=1))  # Y helps X
        gate_yx = self.gate_net(torch.cat([ch_y, ch_x], dim=1))  # X helps Y
        out_x = ch_x + self.alpha * gate_xy * ch_y
        out_y = ch_y + self.alpha * gate_yx * ch_x
        return torch.stack([out_x, out_y], dim=1)


class GeneratorClass(nn.Module):
    def __init__(self, FreqDim,
                 use_cross_channel_fusion=False,
                 use_sinusoidal_pe=False,
                 use_multi_scale_skips=True,
                 use_speech_mask=False,
                 use_snake: bool = False,
                 use_attn_recon: bool = False,
                 use_ffn_dil: bool = False,
                 use_windowed_attn_se: bool = False,
                 use_linear_attn_se: bool = False,
                 use_istft_head: bool = False,
                 use_rephase: bool = False,
                 use_wide_dil: bool = False,
                 denoiser_extra_tscb: int = 0,
                 recon_extra_flash: int = 0):
        super().__init__()
        self.use_rephase = use_rephase
        self.use_cross_channel_fusion = use_cross_channel_fusion
        if use_cross_channel_fusion:
            self.CrossChannelFusion = CrossChannelFusion(n_mel=FreqDim)
        self.Denoiser    = DenoiseClass(extra_tscb=denoiser_extra_tscb)
        self.Reconstruct = ReconstrucClass(
            FreqDim,
            use_sinusoidal_pe=use_sinusoidal_pe,
            use_multi_scale_skips=use_multi_scale_skips,
            use_speech_mask=use_speech_mask,
            use_snake=use_snake,
            use_attn_recon=use_attn_recon,
            use_ffn_dil=use_ffn_dil,
            use_windowed_attn_se=use_windowed_attn_se,
            use_linear_attn_se=use_linear_attn_se,
            use_istft_head=use_istft_head,
            use_wide_dil=use_wide_dil,
            recon_extra_flash=recon_extra_flash)
        # Re-phasing head: keeps the base magnitude, predicts a new phase, iSTFT.
        if use_rephase:
            from .rephase import RephaseHead
            self.RephaseHead = RephaseHead()

    def forward(self, x):  # (B, 2, F, T)
        if self.use_cross_channel_fusion:
            x = self.CrossChannelFusion(x)
        x_denoised, feat0 = self.Denoiser(x)
        y_hat, mask = self.Reconstruct(x_denoised, feat0)
        if self.use_rephase:
            # base path stays as-is; re-phase its output. phase returned via the
            # 3rd slot so the training step can apply the anti-wrapping loss.
            y_out, phase = self.RephaseHead(y_hat)
            return y_out, x_denoised, phase
        return y_hat, x_denoised, mask
