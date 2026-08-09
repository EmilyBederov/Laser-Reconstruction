#!/usr/bin/env python
"""Whisper WER on AudioLaser reconstructions — intelligibility metric.

Reconstructs each validation utterance (chunked exactly like training: wave_len
segments, [X,Y] channels) and runs Whisper to measure how intelligible the
recovered speech is. Reports:
  - WER(reconstructed)  — the number we care about
  - WER(clean)          — oracle ceiling (Whisper on the clean reference)
  - WER(laser X raw)    — floor (Whisper on the raw laser input), optional

Usage:
  python scripts/whisper_wer_recon.py \
      --ckpt checkpoints/new_arch/epoch017-pesq1.498.ckpt \
      --model new_arch \
      --csv data/val.csv \
      --whisper small.en --n 50 --out wer_new_arch.csv
"""
import argparse, sys, warnings, re
from pathlib import Path
import numpy as np, soundfile as sf, torch, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # AudioLaser/
warnings.filterwarnings("ignore")
from hydra import compose, initialize_config_dir
from src.lightning.system import AudioLaserSystem
from src.data.dataset import laser_mel_spec
import whisper, jiwer

WL = 98304  # wave_len (matches training)
SR = 16000

def _norm(s):
    """Lowercase, strip punctuation, collapse whitespace — version-proof WER prep."""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def mel(w, device):
    t = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0).to(device)
    return laser_mel_spec(t, SR, 1024, 128, 80, 8000, 768)


def load_model(model_name, ckpt, device):
    with initialize_config_dir(config_dir=str((Path(__file__).resolve().parent.parent / "configs")), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_name}", f"trainer={model_name}"])
    cfg.data.max_frames = 768
    s = AudioLaserSystem(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = {k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()}
    s.load_state_dict(st, strict=False)
    s.eval().to(device)
    return s


def read_wav(path, n=None):
    a, sr = sf.read(path, dtype="float32", always_2d=False) if n is None else \
            (sf.read(path, start=0, frames=n, dtype="float32", always_2d=False))
    if a.ndim > 1:
        a = a.mean(1)
    return a


@torch.no_grad()
def reconstruct_full(model, x, y, device):
    """Chunk into wave_len segments, reconstruct each, concatenate."""
    L = min(len(x), len(y))
    x, y = x[:L], y[:L]
    out = []
    for st in range(0, L, WL):
        cx = x[st:st + WL]; cy = y[st:st + WL]
        clen = len(cx)
        if clen < WL:  # pad last chunk
            cx = np.pad(cx, (0, WL - clen)); cy = np.pad(cy, (0, WL - clen))
        spec = torch.cat([mel(cx, device), mel(cy, device)], 1)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            wav = model.gen(spec)[0][0, 0].float().cpu().numpy()
        out.append(wav[:clen])
    return np.concatenate(out) if out else np.zeros(1, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, help="hydra model/trainer config name (e.g. new_arch)")
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--n", type=int, default=0, help="limit #utterances (0=all)")
    ap.add_argument("--raw", action="store_true", help="also score raw laser-X (floor)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [whisper] {args.whisper}  [ckpt] {args.ckpt}", flush=True)

    asr = whisper.load_model(args.whisper, device=device)
    model = load_model(args.model, args.ckpt, device)

    df = pd.read_csv(args.csv)
    rows = [r for _, r in df.iterrows() if "channel_x" in str(r["radar_path"])]
    if args.n:
        rows = rows[:args.n]
    print(f"[data] {len(rows)} unique utterances", flush=True)

    refs, hyp_rec, hyp_clean, hyp_raw, records = [], [], [], [], []
    dec = dict(language="en", fp16=(device.type == "cuda"))
    for i, r in enumerate(rows):
        xp = r["radar_path"]; yp = xp.replace("channel_x", "channel_y"); cp = r["audio_path"]
        ref = str(r["text"]).strip()
        if not ref:
            continue
        x = read_wav(xp); y = read_wav(yp); clean = read_wav(cp)
        rec = reconstruct_full(model, x, y, device)

        h_rec = asr.transcribe(rec.astype(np.float32), **dec)["text"].strip()
        h_cln = asr.transcribe(clean.astype(np.float32), **dec)["text"].strip()
        refs.append(ref); hyp_rec.append(h_rec); hyp_clean.append(h_cln)
        rec_row = {"ref": ref, "hyp_recon": h_rec, "hyp_clean": h_cln}
        if args.raw:
            h_raw = asr.transcribe(x.astype(np.float32), **dec)["text"].strip()
            hyp_raw.append(h_raw); rec_row["hyp_raw"] = h_raw
        records.append(rec_row)
        if (i + 1) % 10 == 0:
            w = jiwer.wer([_norm(r) for r in refs], [_norm(h) for h in hyp_rec])
            print(f"  [{i+1}/{len(rows)}] running WER(recon)={w*100:.2f}%", flush=True)

    def W(h): return jiwer.wer([_norm(r) for r in refs], [_norm(x) for x in h]) * 100
    print("\n" + "=" * 50)
    print(f"{'WER reconstructed':<28}{W(hyp_rec):>8.2f} %")
    print(f"{'WER clean (oracle ceiling)':<28}{W(hyp_clean):>8.2f} %")
    if args.raw:
        print(f"{'WER raw laser-X (floor)':<28}{W(hyp_raw):>8.2f} %")
    print("=" * 50)

    if args.out:
        pd.DataFrame(records).to_csv(args.out, index=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
