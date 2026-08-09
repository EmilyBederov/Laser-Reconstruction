#!/usr/bin/env python
"""Consistent PESQ/STOI/SI-SDR eval for an AudioLaser checkpoint over the val set.

Scores per wave_len chunk (mirrors training-time validation), RMS-matches the
output to the reference before metrics (as validation does), and reports the mean
over all chunks. Use to get ONE clean, comparable table across models.

  python scripts/eval_metrics.py --ckpt <ckpt> --model new_arch \
      --csv data/val.csv --out results/metrics_new_arch.csv
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, soundfile as sf, torch, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from hydra import compose, initialize_config_dir
from src.lightning.system import AudioLaserSystem
from src.data.dataset import laser_mel_spec
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn

WL = 98304; SR = 16000


def load(model, ckpt, device):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent.parent / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model}", f"trainer={model}"])
    cfg.data.max_frames = 768
    s = AudioLaserSystem(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = {k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()}
    s.load_state_dict(st, strict=False)
    return s.eval().to(device)


def mel(w, device):
    t = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0).to(device)
    return laser_mel_spec(t, SR, 1024, 128, 80, 8000, 768)


def rd(p):
    a, _ = sf.read(p, dtype="float32", always_2d=False)
    return a if a.ndim == 1 else a.mean(1)


def sisdr(ref, deg):
    n = min(len(ref), len(deg)); ref, deg = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    ref = ref - ref.mean(); deg = deg - deg.mean()
    a = np.dot(deg, ref) / (np.dot(ref, ref) + 1e-9); t = a * ref; e = deg - t
    return 10 * np.log10((np.dot(t, t) + 1e-9) / (np.dot(e, e) + 1e-9))


def score_chunk(ref, deg):
    n = min(len(ref), len(deg)); ref, deg = ref[:n], deg[:n]
    rr = np.sqrt(np.mean(ref**2)) + 1e-8; dd = np.sqrt(np.mean(deg**2)) + 1e-8
    dm = deg * (rr / dd)
    try: p = pesq_fn(SR, ref, dm, "wb")
    except Exception: p = np.nan
    try: st = stoi_fn(ref, dm, SR)
    except Exception: st = np.nan
    return p, st, sisdr(ref, dm)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [ckpt] {args.ckpt}", flush=True)
    model = load(args.model, args.ckpt, device)

    df = pd.read_csv(args.csv)
    rows = [r for _, r in df.iterrows() if "channel_x" in str(r["radar_path"])]
    if args.n: rows = rows[:args.n]
    print(f"[data] {len(rows)} utterances", flush=True)

    allm, rec = [], []
    for i, r in enumerate(rows):
        xp = r["radar_path"]; yp = xp.replace("channel_x", "channel_y"); cp = r["audio_path"]
        x = rd(xp); y = rd(yp); cl = rd(cp)
        L = min(len(x), len(y), len(cl))
        for st0 in range(0, L, WL):
            cx, cy, cc = x[st0:st0+WL], y[st0:st0+WL], cl[st0:st0+WL]
            if len(cx) < WL: break          # skip ragged tail chunk
            if np.abs(cc).max() < 0.01: continue
            spec = torch.cat([mel(cx, device), mel(cy, device)], 1)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                enh = model.gen(spec)[0][0, 0].float().cpu().numpy()
            allm.append(score_chunk(cc, enh))
        if (i + 1) % 25 == 0:
            a = np.array(allm)
            print(f"  [{i+1}/{len(rows)}] PESQ={np.nanmean(a[:,0]):.3f} "
                  f"STOI={np.nanmean(a[:,1]):.3f} SI-SDR={np.nanmean(a[:,2]):.2f}", flush=True)

    a = np.array(allm)
    print("\n" + "=" * 56)
    print(f"{'PESQ':>8}{'STOI':>8}{'SI-SDR':>9}   ({len(a)} chunks)")
    print(f"{np.nanmean(a[:,0]):>8.3f}{np.nanmean(a[:,1]):>8.3f}{np.nanmean(a[:,2]):>9.2f}")
    print("=" * 56)
    if args.out:
        pd.DataFrame(a, columns=["pesq", "stoi", "si_sdr"]).to_csv(args.out, index=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
