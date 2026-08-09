import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _read_chunk(path: str, start: int, n_frames: int) -> torch.Tensor:
    data, _ = sf.read(path, start=start, frames=n_frames, dtype="float32", always_2d=False)
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    if wav.numel() < n_frames:
        wav = F.pad(wav, (0, n_frames - wav.numel()))
    return wav


def laser_mel_spec(wav_b1t: torch.Tensor, sr: int, n_fft: int,
                   hop: int, n_mel: int, fmax: float, max_frames: int) -> torch.Tensor:
    """(B,1,T) → (B,1,n_mel,max_frames) Whisper-style log-mel."""
    B, _, T = wav_b1t.shape
    wav = wav_b1t.squeeze(1)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop,
        win_length=n_fft, n_mels=n_mel, f_min=0.0, f_max=fmax, power=2.0,
    ).to(wav.device)
    mel = mel_tf(wav)[..., :max_frames]
    log_mel = torch.clamp(mel, min=1e-10).log10()
    max_val = log_mel.reshape(B, -1).max(dim=1)[0].view(B, 1, 1)
    log_mel = torch.maximum(log_mel, max_val - 8.0)
    return ((log_mel + 4.0) / 4.0).unsqueeze(1)


class CsvLaserDataset(torch.utils.data.Dataset):
    """Paired laser/clean dataset from mmradar CSV manifest.
    Each recording is split into non-overlapping WAVE_LEN-sample chunks."""

    def __init__(self, csv_path: str, sr: int, n_fft: int, hop: int,
                 n_mel: int, fmax: float, max_frames: int, train: bool = True,
                 silence_thresh: float = 0.01):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop        = hop
        self.n_mel      = n_mel
        self.fmax       = fmax
        self.max_frames = max_frames
        self.wave_len   = hop * max_frames
        self.train      = train

        df = pd.read_csv(csv_path)
        all_pairs = list(zip(df["radar_path"], df["audio_path"]))

        self.chunks = []
        skipped = 0
        for x_path, y_path in all_pairs:
            try:
                ix = sf.info(x_path)
                iy = sf.info(y_path)
            except Exception:
                skipped += 1
                continue
            if ix.samplerate != sr or iy.samplerate != sr:
                skipped += 1
                continue
            n_frames = min(ix.frames, iy.frames)
            for i in range(max(1, n_frames // self.wave_len)):
                self.chunks.append((x_path, y_path, i * self.wave_len))

        if skipped:
            print(f"[CsvLaserDataset] skipped {skipped} bad pairs ({csv_path})")
        print(f"[CsvLaserDataset] {len(self.chunks)} chunks from {len(all_pairs)-skipped} recordings")
        if not self.chunks:
            raise RuntimeError(f"No valid chunks in {csv_path}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        x_path, y_path, start = self.chunks[idx]
        if self.train:
            start = max(0, start + random.randint(-200, 200))

        x_wav = _read_chunk(x_path, start, self.wave_len)
        y_wav = _read_chunk(y_path, start, self.wave_len)

        # Resolve both X and Y laser channel paths regardless of which row is primary.
        # channel_x rows: path_x == x_path (no reload), path_y = companion Y
        # channel_y rows: path_y == x_path (no reload), path_x = companion X
        path_x = x_path.replace("channel_y", "channel_x")
        path_y = x_path.replace("channel_x", "channel_y")

        las_x = x_wav if path_x == x_path else (
            _read_chunk(path_x, start, self.wave_len) if os.path.exists(path_x) else x_wav.clone()
        )
        las_y = x_wav if path_y == x_path else (
            _read_chunk(path_y, start, self.wave_len) if os.path.exists(path_y) else x_wav.clone()
        )

        def _mel(w):
            return laser_mel_spec(w.unsqueeze(0).unsqueeze(0),
                                  self.sr, self.n_fft, self.hop,
                                  self.n_mel, self.fmax, self.max_frames).squeeze(0)

        # Channel 0 = X, channel 1 = Y (consistent across all rows)
        x_spec = torch.cat([_mel(las_x), _mel(las_y)], dim=0)  # (2, n_mel, T)

        # Channel strategy: 50% both, 25% X-only (zero Y), 25% Y-only (zero X)
        if self.train:
            r = random.random()
            if 0.50 <= r < 0.75:
                x_spec[1] = 0.0   # X only
            elif r >= 0.75:
                x_spec[0] = 0.0   # Y only

        y_spec  = _mel(y_wav)                              # (1, n_mel, T)
        y_wav_t = y_wav[:self.wave_len].unsqueeze(0)       # (1, wave_len)
        x_wav_t = x_wav[:self.wave_len].unsqueeze(0)       # (1, wave_len) for monitoring

        return x_spec, y_spec, y_wav_t, x_wav_t, Path(x_path).stem


def collate_fn(batch):
    x_specs, y_specs, y_wavs, x_wavs, names = zip(*batch)
    return (torch.stack(x_specs), torch.stack(y_specs),
            torch.stack(y_wavs), torch.stack(x_wavs), list(names))


class LaserDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _make_ds(self, split):
        d = self.cfg.data
        return CsvLaserDataset(
            csv_path        = d.train_csv if split == "train" else d.val_csv,
            sr              = d.sr,
            n_fft           = d.n_fft_mel,
            hop             = d.hop_length_mel,
            n_mel           = d.n_mel,
            fmax            = d.fmax,
            max_frames      = d.max_frames,
            train           = split == "train",
            silence_thresh  = d.get("silence_threshold", 0.01),
        )

    def setup(self, stage=None):
        self.train_ds = self._make_ds("train")
        self.val_ds   = self._make_ds("val")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.cfg.data.batch_size,
            shuffle     = True,
            num_workers = self.cfg.data.num_workers,
            pin_memory  = True,
            drop_last   = True,
            persistent_workers = True,
            collate_fn  = collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.cfg.data.batch_size,
            shuffle     = False,
            num_workers = self.cfg.data.num_workers,
            pin_memory  = True,
            drop_last   = False,
            persistent_workers = True,
            collate_fn  = collate_fn,
        )
