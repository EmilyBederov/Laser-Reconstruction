# AudioLaser

Speech **reconstruction** from laser Doppler vibrometry (LDV). A model that takes the
two orthogonal laser measurement axes of a vibrating surface and reconstructs the
speech waveform, trained adversarially with multi-resolution spectral and
complex-STFT (phase) losses.

```
laser (X, Y)  ──►  log-mel  ──►  CrossChannelFusion ──► Denoiser ──► Reconstructor ──►  speech
                                        (2-ch)         (2-D conf.)   (1-D HiFi-GAN)
```

---

## Install

Tested with **Python 3.10, CUDA 11.8, NVIDIA A100**. Training uses `bf16-mixed`
precision, which requires an **Ampere or newer GPU** (A100 / L40S / H100 / H200);
on older GPUs, lower the precision in the trainer config. Inference scripts fall
back to FP32 automatically on CPU.

```bash
conda create -n audiolaser python=3.10 -y
conda activate audiolaser
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Repository layout

```
audiolaser/
├── train.py                     # training entry point (Hydra)
├── configs/
│   ├── config.yaml              # top-level defaults
│   ├── data/default.yaml        # data + spectrogram settings, csv paths
│   ├── model/                   # audiolaser, base_attn, phase_riloss
│   └── trainer/                 # matching training recipes
├── src/
│   ├── models/                  # generator, denoiser, reconstructor, rephase, discriminator, …
│   ├── lightning/system.py      # LightningModule: losses, optim, validation
│   ├── data/dataset.py          # paired laser/clean dataset + log-mel
│   └── loss/losses.py           # phase (complex-STFT / anti-wrap) losses
├── scripts/
│   ├── reconstruct.py           # RUN: laser wav(s) -> reconstructed speech
│   ├── eval_metrics.py          # PESQ / STOI / SI-SDR over a manifest
│   └── whisper_wer_recon.py     # intelligibility: Whisper WER on reconstructions
└── data/example.csv             # manifest format
```

## Data format

A CSV manifest with three columns. Each utterance has **two** laser rows (one per
axis); the companion axis is found by swapping `channel_x`/`channel_y` in the path.

```csv
radar_path,audio_path,text
data/laser/utt0001_channel_x.wav,data/clean/utt0001.wav,the reference transcript
data/laser/utt0001_channel_y.wav,data/clean/utt0001.wav,the reference transcript
```

- `radar_path`  — laser waveform (16 kHz mono), path contains `channel_x` or `channel_y`
- `audio_path`  — clean close-talk reference (16 kHz mono)
- `text`        — reference transcript (used only for WER evaluation)

Point `configs/data/default.yaml` (`train_csv`, `val_csv`) at your manifests.

## Train

```bash
# main model (trains from scratch)
python train.py                                   # model=audiolaser

# attention baseline
python train.py model=base_attn   trainer=base_attn

# RI-loss phase variant (fine-tunes the audiolaser checkpoint)
python train.py model=phase_riloss trainer=phase_riloss

# no Weights & Biases:
WANDB_MODE=disabled python train.py
# multi-GPU (torchrun sets WORLD_SIZE):
torchrun --nproc_per_node=3 train.py
```

Checkpoints go to `checkpoints/<model>/` (`last.ckpt` + top-3 by validation PESQ);
training auto-resumes from `last.ckpt`.

## Evaluate

```bash
# perceptual quality + intelligibility metrics over the val manifest
python scripts/eval_metrics.py --ckpt checkpoints/audiolaser/last.ckpt \
    --model audiolaser --csv data/val.csv --out results/metrics.csv

# Whisper WER on the reconstructions (+ clean ceiling and raw-laser floor)
python scripts/whisper_wer_recon.py --ckpt checkpoints/audiolaser/last.ckpt \
    --model audiolaser --csv data/val.csv --whisper small.en --raw \
    --out results/wer.csv
```

## Run (inference)

```bash
python scripts/reconstruct.py --ckpt checkpoints/audiolaser/last.ckpt \
    --model audiolaser --x laser_x.wav --y laser_y.wav --out reconstructed.wav
```
If only one axis is available, pass `--x laser.wav` (it is used for both channels).

## Models

| config | description |
|---|---|
| **`audiolaser`** | main model: cross-channel fusion + dilated Conformer denoiser + HiFi-GAN reconstructor (`use_ffn_dil`) |
| `phase_riloss` | `audiolaser` fine-tuned with a strong complex-STFT (real/imaginary) phase loss |
| `base_attn` | attention baseline (mmSpeech-style) |

**Losses** (`src/lightning/system.py`): multi-scale mel + multi-resolution STFT
(magnitude) + MPD/MRD adversarial + feature matching, plus optional complex-STFT
(`lambda_phase`) and MP-SENet anti-wrapping (`lambda_phase_aw`) phase supervision.

## Key config knobs

`configs/data/default.yaml`: `sr` 16000, `n_mel` 80, `n_fft_mel` 1024,
`hop_length_mel` 128, `max_frames` 768, `batch_size`.
`configs/trainer/*.yaml`: `epochs`, `lr`, loss weights (`lambda_mel`, `lambda_mrstft`,
`lambda_phase`, `lambda_adv`, `lambda_fm`), `ckpt_dir`, `warmstart_path`.
