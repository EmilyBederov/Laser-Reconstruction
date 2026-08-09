import torch
import torch.nn.functional as F
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

import typing
from typing import Optional, List, Union, Dict, Tuple
from collections import namedtuple
import math
import functools

import numpy as np
import torch
from scipy import signal

class PhaseAntiWrapLoss(nn.Module):
    """MP-SENet anti-wrapping phase losses: IP + GD + IAF.

    Phase wraps at ±π, so a raw L1 on phase explodes at discontinuities. The
    anti-wrapping function f(x) = |x − 2π·round(x/2π)| maps any phase *difference*
    back into [−π, π], giving a smooth target. Three terms (Ai et al., MP-SENet):
      IP  — instantaneous phase:        f(φ̂ − φ)
      GD  — group delay (Δ along freq): f(Δ_f φ̂ − Δ_f φ)
      IAF — inst. angular freq (Δ time):f(Δ_t φ̂ − Δ_t φ)

    Compares a predicted phase (B, F, T) against the clean reference waveform.
    n_fft / hop must match the head that produced the predicted phase.
    """
    def __init__(self, n_fft: int = 1024, hop: int = 128):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _aw(x):                                  # anti-wrapping function → [−π, π]
        return torch.abs(x - 2 * math.pi * torch.round(x / (2 * math.pi)))

    def forward(self, phase_pred: torch.Tensor, y_clean: torch.Tensor) -> torch.Tensor:
        """phase_pred: (B, F, T).  y_clean: (B, 1, T_wav)."""
        S = torch.stft(y_clean[:, 0].float(), self.n_fft, self.hop, self.n_fft,
                       self.win.to(y_clean.device), center=True, return_complex=True)
        phase_clean = torch.angle(S)             # (B, F, T)
        T = min(phase_pred.shape[-1], phase_clean.shape[-1])
        p, c = phase_pred[..., :T], phase_clean[..., :T]

        ip  = self._aw(p - c).mean()
        gd  = self._aw(torch.diff(p, dim=-2) - torch.diff(c, dim=-2)).mean()   # Δ freq
        iaf = self._aw(torch.diff(p, dim=-1) - torch.diff(c, dim=-1)).mean()   # Δ time
        return ip + gd + iaf


class MultiResolutionComplexSTFTLoss(nn.Module):
    """L1 on REAL and IMAGINARY parts of the STFT at multiple resolutions.

    Unlike MultiResolutionSTFTLoss (which uses |STFT| and is phase-blind), the
    real/imaginary parts depend on phase, so minimizing this loss REQUIRES correct
    phase. This is the explicit phase-supervision signal that the magnitude-only
    losses (mel, mrstft, mag) cannot provide. Standard "RI loss" from speech
    enhancement / neural vocoders.

    resolutions: list of (n_fft, hop_length, win_length)
    """
    def __init__(self, resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B, 1, T) float32 waveforms."""
        loss = 0.0
        B, C, T = x.shape
        for n_fft, hop, win_len in self.resolutions:
            window = torch.hann_window(win_len, device=x.device)
            X = torch.stft(x.reshape(B * C, T), n_fft=n_fft, hop_length=hop,
                           win_length=win_len, window=window, return_complex=True, center=True)
            Y = torch.stft(y.reshape(B * C, T), n_fft=n_fft, hop_length=hop,
                           win_length=win_len, window=window, return_complex=True, center=True)
            loss = loss + F.l1_loss(X.real, Y.real) + F.l1_loss(X.imag, Y.imag)
        return loss / len(self.resolutions)


class MultiResolutionSTFTLoss(nn.Module):
    """L1 on log-magnitude raw STFT at multiple resolutions.
    Complements MultiScaleMelSpectrogramLoss: mel warping compresses frequencies
    above ~4kHz, so fine high-freq detail (consonants, sibilants) is only supervised here.

    resolutions: list of (n_fft, hop_length, win_length)
    Loss is averaged across resolutions so the weight stays scale-independent.
    """
    def __init__(
        self,
        resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)),
        clamp_eps: float = 1e-5,
    ):
        super().__init__()
        self.resolutions = resolutions
        self.clamp_eps = clamp_eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B, 1, T) float32 waveforms in [-1, 1]"""
        loss = 0.0
        B, C, T = x.shape
        for n_fft, hop, win_len in self.resolutions:
            window = torch.hann_window(win_len, device=x.device)
            x_stft = torch.stft(
                x.reshape(B * C, T), n_fft=n_fft, hop_length=hop,
                win_length=win_len, window=window, return_complex=True, center=True,
            )
            y_stft = torch.stft(
                y.reshape(B * C, T), n_fft=n_fft, hop_length=hop,
                win_length=win_len, window=window, return_complex=True, center=True,
            )
            x_log_mag = x_stft.abs().clamp(min=self.clamp_eps).log()
            y_log_mag = y_stft.abs().clamp(min=self.clamp_eps).log()
            loss = loss + F.l1_loss(x_log_mag, y_log_mag)
        return loss / len(self.resolutions)


#Discriminator losss
# INPUT: disc_real_outputs = list of tensors, each (B, N_i) or (B, *)
#        disc_generated_outputs = list of tensors, each (B, N_i) or (B, *)
# OUTPUT: loss (scalar tensor), r_losses (list of python floats), g_losses (list of python floats)
def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


#L_adv
# INPUT: disc_outputs = list of tensors (scores on generated audio), each (B, N_i) or (B, *)
# OUTPUT: loss (scalar tensor), gen_losses (list of scalar tensors, one per sub-discriminator)
def generator_loss_Ladv(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:

    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


#L_fm
# INPUT: x = estimated waveform tensor (B, 1, T)
#        y = reference waveform tensor (B, 1, T)
# OUTPUT: loss (scalar tensor)
def generator_loss_feature_loss(
    fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
) -> torch.Tensor:

    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2  # This equates to lambda=2.0 for the feature matching loss



#L_mel
# INPUT: x = estimated waveform tensor (B, 1, T)
#        y = reference waveform tensor (B, 1, T)
# OUTPUT: loss (scalar tensor)
class MultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
        hf_emphasis: float = 0.0,  # 0=uniform bin weighting (default), >0 boosts high freqs
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.hf_emphasis = float(hf_emphasis)

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from:
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            if self.hf_emphasis > 0:
                # Per-bin weight: w_b = 1 + hf_emphasis * (b / (N_mel-1))
                # bin 0 → 1.0, top bin → 1 + hf_emphasis. Default hf_emphasis=0 → uniform.
                Nb = x_logmels.size(-2)
                bin_w = 1.0 + self.hf_emphasis * (
                    torch.arange(Nb, device=x_logmels.device, dtype=x_logmels.dtype) / max(Nb - 1, 1)
                )
                bin_w = bin_w.view(1, 1, Nb, 1)  # (1, 1, n_mels, T)
                diff_log = (x_logmels - y_logmels).abs() * bin_w
                loss += self.log_weight * diff_log.mean()
                loss += self.mag_weight * diff_log.mean()  # mirror existing dual-add (kept for parity)
            else:
                loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
                loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss