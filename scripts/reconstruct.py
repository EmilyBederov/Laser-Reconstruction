#!/usr/bin/env python
"""Run AudioLaser on a laser recording: laser waveform(s) -> reconstructed speech.

The model takes the two orthogonal laser axes (channel X and channel Y). Give both
with --x/--y; if only --x is given it is used for both channels. Audio is processed
in fixed windows and the outputs are concatenated to reconstruct the full utterance.

  python scripts/reconstruct.py \
      --ckpt checkpoints/audiolaser/last.ckpt --model audiolaser \
      --x laser_x.wav --y laser_y.wav --out reconstructed.wav
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from hydra import compose, initialize_config_dir
from src.lightning.system import AudioLaserSystem
from src.data.dataset import laser_mel_spec

WL = 98304; SR = 16000


def load(model_name, ckpt, device):
    cfgdir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_name}", f"trainer={model_name}"])
    cfg.data.max_frames = 768
    s = AudioLaserSystem(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = {k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()}
    res = s.load_state_dict(st, strict=False)
    if res.missing_keys:
        print(f"[warn] {len(res.missing_keys)} params not found in checkpoint (left at init)")
    return s.eval().to(device), cfg


def mel(w, cfg, device):
    t = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).unsqueeze(0).to(device)
    d = cfg.data
    return laser_mel_spec(t, d.sr, d.n_fft_mel, d.hop_length_mel, d.n_mel, d.fmax, d.max_frames)


def rd(p):
    a, _ = sf.read(p, dtype="float32", always_2d=False)
    return a if a.ndim == 1 else a.mean(1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="audiolaser")
    ap.add_argument("--x", required=True, help="laser channel-X waveform")
    ap.add_argument("--y", default=None, help="laser channel-Y waveform (defaults to X)")
    ap.add_argument("--out", default="reconstructed.wav")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [ckpt] {args.ckpt}", flush=True)
    model, cfg = load(args.model, args.ckpt, device)

    x = rd(args.x)
    y = rd(args.y) if args.y else x
    L = min(len(x), len(y))
    out = []
    for st0 in range(0, L, WL):
        cx, cy = x[st0:st0+WL], y[st0:st0+WL]
        pad = WL - len(cx)
        if pad > 0:
            cx = np.pad(cx, (0, pad)); cy = np.pad(cy, (0, pad))
        spec = torch.cat([mel(cx, cfg, device), mel(cy, cfg, device)], 1)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            enh = model.gen(spec)[0][0, 0].float().cpu().numpy()
        out.append(enh[:WL - pad] if pad > 0 else enh)

    out = np.concatenate(out)
    out = out / (np.abs(out).max() + 1e-8) * 0.95   # peak-normalise
    sf.write(args.out, out.astype(np.float32), SR)
    print(f"[saved] {args.out}  ({len(out)/SR:.1f}s)")


if __name__ == "__main__":
    main()
