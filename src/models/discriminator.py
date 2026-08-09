import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Tuple


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        h: AttrDict,
        period: int, #I CHANGED HERE
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = int(period) #I CHANGED HERE
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        int(32 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(32 * self.d_mult),
                        int(128 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(128 * self.d_mult),
                        int(512 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(512 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(1024 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        1,
                        padding=(2, 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h: AttrDict):
        super().__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print(f"mpd_reshapes: {self.mpd_reshapes}")
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm)
                for rs in self.mpd_reshapes
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[int]):
        super().__init__()

        self.resolution = resolution
        assert (
            len(self.resolution) == 3
        ), f"MRD layer requires list with len=3, got {self.resolution}"
        self.lrelu_slope = 0.1

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print(
                f"[INFO] overriding MRD use_spectral_norm as {cfg.mrd_use_spectral_norm}"
            )
            norm_f = (
                weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
            )
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print(f"[INFO] overriding mrd channel multiplier as {cfg.mrd_channel_mult}")
            self.d_mult = cfg.mrd_channel_mult

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution

        # STFT must be FP32: cuFFT doesn't support BF16
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()

            x = F.pad(
                x,
                (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
                mode="reflect",
            )
            x = x.squeeze(1)

            window = torch.hann_window(win_length, device=x.device, dtype=torch.float32)

            x = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            )

            x = torch.view_as_real(x)  # [B, F, TT, 2]
            mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert (
            len(self.resolutions) == 3
        ), f"MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}"
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HighBandMRD(nn.Module):
    """High-pass-filtered MRD: scores only the >cutoff Hz band so the
    discriminator focuses adversarial penalty on high-frequency content.
    Wraps a standard MultiResolutionDiscriminator without modifying it.
    Only instantiated when use_hf_mrd=True so default training is unaffected."""
    def __init__(self, cfg, cutoff_hz: float = 2000.0, sr: int = 16000):
        super().__init__()
        self.mrd = MultiResolutionDiscriminator(cfg)
        self.cutoff_hz = float(cutoff_hz)
        self.sr        = int(sr)

    def _hpf(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T). torchaudio's biquad expects (..., T) — keep last dim.
        return TAF.highpass_biquad(x, self.sr, self.cutoff_hz)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return self.mrd(self._hpf(y), self._hpf(y_hat))


class PQMF(nn.Module):
    """Pseudo-Quadrature Mirror Filter bank — analysis only.

    Splits (B, 1, T) → (B, subbands, T // subbands).
    Based on Multi-Band MelGAN (Yamamoto et al., 2020): Kaiser-windowed prototype
    low-pass filter, modulated into K band-pass filters.
    """
    def __init__(self, subbands: int = 4, taps: int = 62, beta: float = 9.0):
        super().__init__()
        self.subbands = subbands
        n = np.arange(taps + 1, dtype=np.float64)
        cutoff = 1.0 / subbands
        h = cutoff * np.sinc(cutoff * (n - taps / 2))
        h *= np.kaiser(taps + 1, beta)
        h /= h.sum()
        f = np.zeros((subbands, 1, taps + 1), dtype=np.float32)
        for k in range(subbands):
            phase = (2 * k + 1) * np.pi / (2 * subbands)
            f[k, 0] = (2 * h * np.cos(phase * (n - taps / 2) + (-1) ** k * np.pi / 4)).astype(np.float32)
        self.register_buffer("filter", torch.from_numpy(f))
        self.taps = taps

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, subbands, T // subbands)"""
        return F.conv1d(
            F.pad(x, (self.taps // 2, self.taps // 2)),
            self.filter,
            stride=self.subbands,
        )


class SubBandDiscriminator(nn.Module):
    """Sub-band discriminator: PQMF analysis + shared MRD with per-band weighting.

    Splits both real and generated waveforms into K=subbands frequency bands via
    PQMF, runs the shared MRD on each band separately, and applies per-band weights.

    Default weights [1, 2, 3, 4] put increasing pressure on high-frequency bands
    (which are hardest to reconstruct from laser input) — directly targets STOI.

    Weight scaling:
      - FM loss (L1):     weight applied to feature maps → loss scales exactly by w
      - Adv loss (LS-GAN): scores scaled by sqrt(w) → loss scales approx by w
    """
    def __init__(self, cfg, subbands: int = 4, taps: int = 62, beta: float = 9.0,
                 band_weights: List[float] = None):
        super().__init__()
        self.subbands = subbands
        if band_weights is None:
            band_weights = [float(k + 1) for k in range(subbands)]
        self.register_buffer("band_weights", torch.tensor(band_weights, dtype=torch.float32))
        self.pqmf = PQMF(subbands, taps, beta=beta)
        sb_cfg = AttrDict(cfg)
        sb_cfg.resolutions = [
            [max(n // subbands, 64), max(h // subbands, 16), max(w // subbands, 64)]
            for n, h, w in cfg.resolutions
        ]
        self.mrd = MultiResolutionDiscriminator(sb_cfg)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_sb     = self.pqmf.analysis(y)      # (B, K, T//K)
        y_hat_sb = self.pqmf.analysis(y_hat)
        real_all, fake_all, fmap_r_all, fmap_g_all = [], [], [], []
        for k in range(self.subbands):
            w = self.band_weights[k].item()
            r_k, f_k, fmr_k, fmg_k = self.mrd(y_sb[:, k:k+1], y_hat_sb[:, k:k+1])
            ws = w ** 0.5  # sqrt(w): scales LS-GAN adv loss ≈ by w
            real_all.extend([s * ws for s in r_k])
            fake_all.extend([s * ws for s in f_k])
            fmap_r_all.extend([[f * w for f in fm] for fm in fmr_k])   # exactly scales FM loss by w
            fmap_g_all.extend([[f * w for f in fm] for fm in fmg_k])
        return real_all, fake_all, fmap_r_all, fmap_g_all


# ============================================================================
#  MelDiscriminator — 2D PatchGAN over the MEL SPECTROGRAM (magnitude domain).
#  Motivation: BigVGAN proved clean-mel -> 4.26 PESQ while our-mel -> 1.53, i.e.
#  the bottleneck is MAGNITUDE, not phase. L1/mel losses reward the *average*
#  plausible mel -> blurry/soft prediction. This adversarial signal on the mel
#  pushes the predicted magnitude to look like a real SHARP mel (formants /
#  harmonics), fixing the diffuse softness the mel-diagnostic revealed.
#  Operates on [B,1,F,T]: real = clean mel (y_spec), fake = predicted mel
#  (x_mag_pred, the denoiser output). Returns (rs, gs, fmap_rs, fmap_gs) lists
#  exactly like MPD/MRD so the existing discriminator/generator loss helpers
#  apply unchanged. Training-only (no inference cost). Default OFF.
# ============================================================================
class _MelSubDisc(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1,      ch,   (3, 9), (1, 2), (1, 4))),
            weight_norm(Conv2d(ch,     2*ch, (3, 9), (2, 2), (1, 4))),
            weight_norm(Conv2d(2*ch,   4*ch, (3, 9), (2, 2), (1, 4))),
            weight_norm(Conv2d(4*ch,   4*ch, (3, 3), (1, 1), (1, 1))),
        ])
        self.post = weight_norm(Conv2d(4*ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MelDiscriminator(nn.Module):
    """Multi-scale 2D PatchGAN over the mel spectrogram. scales=2 -> full + time/2."""
    def __init__(self, ch: int = 32, scales: int = 2):
        super().__init__()
        self.discs = nn.ModuleList([_MelSubDisc(ch) for _ in range(scales)])
        self.pool = nn.AvgPool2d((1, 2))   # downsample time for higher scales

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
            List[torch.Tensor], List[torch.Tensor], List, List]:
        # y, y_hat: [B,1,F,T] (real=clean mel, fake=predicted mel)
        if y.dim() == 3:      # [B,F,T] -> [B,1,F,T]
            y = y.unsqueeze(1)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)
        rs, gs, frs, fgs = [], [], [], []
        yr, yg = y, y_hat
        for i, d in enumerate(self.discs):
            if i > 0:
                yr = self.pool(yr); yg = self.pool(yg)
            r, fr = d(yr); g, fg = d(yg)
            rs.append(r); gs.append(g); frs.append(fr); fgs.append(fg)
        return rs, gs, frs, fgs
