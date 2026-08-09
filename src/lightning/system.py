import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pesq import pesq as pesq_fn; HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
try:
    from pystoi import stoi as stoi_fn; HAS_STOI = True
except ImportError:
    HAS_STOI = False
HAS_MOS = False  # DNSMOS disabled: onnxruntime inference takes ~8s/batch, kills val speed

from src.models import GeneratorClass, AttrDict, MultiPeriodDiscriminator, MultiResolutionDiscriminator
from src.loss import (discriminator_loss, generator_loss_Ladv,
                      generator_loss_feature_loss,
                      MultiScaleMelSpectrogramLoss, MultiResolutionSTFTLoss,
                      MultiResolutionComplexSTFTLoss)
from src.data.dataset import laser_mel_spec


def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if hasattr(m, "weight") and m.weight is not None: nn.init.ones_(m.weight)
        if hasattr(m, "bias")   and m.bias   is not None: nn.init.zeros_(m.bias)


def _small_output_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)) and m.out_channels == 1:
        nn.init.normal_(m.weight, 0.0, 0.01)
        if m.bias is not None: nn.init.zeros_(m.bias)


class AudioLaserSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False  # manual GAN update

        m = cfg.model
        self.gen = GeneratorClass(
            FreqDim                  = cfg.data.n_mel,
            use_cross_channel_fusion = m.use_cross_channel_fusion,
            use_sinusoidal_pe        = m.use_sinusoidal_pe,
            use_multi_scale_skips    = m.use_multi_scale_skips,
            use_speech_mask          = m.use_speech_mask,
            use_snake                = getattr(m, "use_snake", False),
            use_attn_recon           = getattr(m, "use_attn_recon", False),
            use_ffn_dil              = getattr(m, "use_ffn_dil", False),
            use_windowed_attn_se     = getattr(m, "use_windowed_attn_se", False),
            use_linear_attn_se       = getattr(m, "use_linear_attn_se", False),
            use_istft_head           = getattr(m, "use_istft_head", False),
            use_wide_dil             = getattr(m, "use_wide_dil", False),
            use_rephase              = getattr(m, "use_rephase", False),
            denoiser_extra_tscb      = int(getattr(m, "denoiser_extra_tscb", 0)),
            recon_extra_flash        = int(getattr(m, "recon_extra_flash", 0)),
        )
        self.gen.apply(_init_weights)
        self.gen.apply(_small_output_init)

        h = AttrDict(
            use_spectral_norm         = False,
            discriminator_channel_mult= m.disc_channel_mult,
            mpd_reshapes              = list(m.mpd_reshapes),
            resolutions               = [list(r) for r in m.mrd_resolutions],
        )
        self.mpd = MultiPeriodDiscriminator(h)
        self.mrd = MultiResolutionDiscriminator(h)
        self.mpd.apply(_init_weights)
        self.mrd.apply(_init_weights)

        # optional high-band MRD: only instantiated when explicitly enabled
        self.use_hf_mrd = bool(getattr(m, "use_hf_mrd", False))
        if self.use_hf_mrd:
            from src.models.discriminator import HighBandMRD
            self.mrd_hf = HighBandMRD(
                h, cutoff_hz=float(getattr(m, "hf_mrd_cutoff_hz", 2000.0)),
                sr=cfg.data.sr,
            )
            self.mrd_hf.apply(_init_weights)

        # sub-band discriminator: PQMF splits signal into K bands, shared MRD with per-band weights
        self.use_subb_disc = bool(getattr(m, "use_subb_disc", False))
        if self.use_subb_disc:
            from src.models.discriminator import SubBandDiscriminator
            subbands = int(getattr(m, "subb_disc_bands", 4))
            raw_w = getattr(m, "subb_disc_weights", None)
            band_weights = list(raw_w) if raw_w is not None else None
            self.subb_disc = SubBandDiscriminator(
                h, subbands=subbands, band_weights=band_weights,
            )
            self.subb_disc.apply(_init_weights)

        # mel-domain discriminator: adversarial sharpening of the PREDICTED MEL
        # (magnitude bottleneck — see BigVGAN diagnostic). real=clean mel (y_spec),
        # fake=denoiser output (x_mag_pred). Training-only, default OFF.
        self.use_mel_disc = bool(getattr(m, "use_mel_disc", False))
        if self.use_mel_disc:
            from src.models.discriminator import MelDiscriminator
            self.mel_disc = MelDiscriminator(
                ch=int(getattr(m, "mel_disc_ch", 32)),
                scales=int(getattr(m, "mel_disc_scales", 2)),
            )
            self.mel_disc.apply(_init_weights)

        # mel loss with optional high-freq emphasis
        self.mel_loss_fn   = MultiScaleMelSpectrogramLoss(
            sampling_rate=cfg.data.sr,
            hf_emphasis=float(getattr(cfg.trainer, "hf_emphasis", 0.0)),
        )
        _mrstft_res = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]
        if getattr(cfg.trainer, "mrstft_short_window", False):
            _mrstft_res = [(128, 32, 128)] + _mrstft_res   # 8ms window: supervises plosive bursts
        self.mrstft_loss_fn = MultiResolutionSTFTLoss(resolutions=_mrstft_res)
        # Explicit phase supervision (real+imag STFT). Only built when enabled.
        self.cstft_loss_fn   = MultiResolutionComplexSTFTLoss(
            resolutions=[(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)])
        # MP-SENet anti-wrapping phase loss (re-phasing head)
        from src.loss import PhaseAntiWrapLoss
        self.phase_aw_loss_fn = PhaseAntiWrapLoss(n_fft=1024, hop=128)
        self.l1 = nn.L1Loss()

        t = cfg.trainer
        self.lam_adv         = float(getattr(t, "lambda_adv", 1.0))  # phase/realism signal; default 1.0 (legacy)
        self.lam_phase       = float(getattr(t, "lambda_phase", 0.0))  # complex-STFT (RI) loss; 0=off
        self.lam_phase_aw    = float(getattr(t, "lambda_phase_aw", 0.0))  # anti-wrap phase loss; 0=off
        self.lam_mel_adv     = float(getattr(t, "lambda_mel_adv", 0.0))  # mel-disc adv+fm weight; 0=off
        self.use_rephase     = bool(getattr(m, "use_rephase", False))
        self.lam_mel         = t.lambda_mel
        self.lam_fm          = t.lambda_fm
        self.lam_mag         = t.lambda_mag
        self.lam_mrstft      = t.lambda_mrstft
        self.lam_mel_consist = t.lambda_mel_consist
        self.lam_energy      = t.lambda_energy
        self.lam_mask        = t.lambda_mask
        self.silence_thresh  = t.silence_threshold

        self.use_vad_input_masking = bool(getattr(cfg.trainer, "use_vad_input_masking", False))
        self.vad_input_thresh      = float(getattr(cfg.trainer, "vad_input_thresh", 0.15))
        # Expand VAD mask by this many frames on each side (150ms @ hop=128, sr=16k → 19 frames)
        _margin_ms = float(getattr(cfg.trainer, "vad_margin_ms", 150.0))
        self.vad_margin_frames = max(0, int(_margin_ms / (cfg.data.hop_length_mel / cfg.data.sr * 1000)))

        self._val_pesq, self._val_stoi, self._val_sisdr = [], [], []
        self._val_silence_rms = []
        self._target_rec_names = []   # discovered on first val epoch, fixed thereafter
        self._rec_buffers      = {}   # name → {y_hat, y_wav, x_wav, specs}

        # ── Two-stage training: warm-start from a PL checkpoint of a good
        #    magnitude model, then add phase supervision. strict=False so new
        #    modules (e.g. the iSTFT head) init fresh while denoiser/CCF/recon
        #    load. If PL later auto-resumes from last.ckpt, that overwrites this
        #    (correct — continue the warm-started run). ─────────────────────────
        warmstart = cfg.trainer.get("warmstart_path", None)
        if warmstart and os.path.isfile(warmstart):
            ck = torch.load(warmstart, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            # strict=False ignores missing/unexpected keys but STILL errors on
            # shape mismatches for shared keys. The warm-start ckpt may differ in
            # discriminator width (disc_channel_mult) or reconstructor head (iSTFT
            # vs upsample stack), so drop any key whose shape doesn't match — those
            # modules then init fresh (intended: new disc / new head trains from 0).
            model_sd = self.state_dict()
            dropped = [k for k, v in sd.items()
                       if k in model_sd and v.shape != model_sd[k].shape]
            sd = {k: v for k, v in sd.items()
                  if k not in dropped and k in model_sd}
            missing, unexpected = self.load_state_dict(sd, strict=False)
            gen_loaded = sum(1 for k in sd if k.startswith("gen."))
            print(f"[WARMSTART] {warmstart}\n"
                  f"[WARMSTART]   loaded {gen_loaded} gen tensors, "
                  f"dropped {len(dropped)} shape-mismatched (fresh), "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        elif warmstart:
            print(f"[WARMSTART] WARNING: {warmstart} not found — training from scratch")

        # ── Optional: freeze the denoiser (+CCF) so only the new reconstruction
        #    head trains. Done here (before configure_optimizers) so the optimizer
        #    only sees trainable params. Decouples phase-head learning from a
        #    fixed, good magnitude front-end (Vocos / mmradar frozen-encoder style).
        if bool(cfg.trainer.get("freeze_denoiser", False)):
            n = 0
            for p in self.gen.Denoiser.parameters():
                p.requires_grad = False; n += 1
            if getattr(self.gen, "use_cross_channel_fusion", False):
                for p in self.gen.CrossChannelFusion.parameters():
                    p.requires_grad = False; n += 1
            print(f"[WARMSTART] froze denoiser(+CCF): {n} param tensors frozen")

        # ── Re-phasing: freeze the ENTIRE base (CCF+Denoiser+Reconstructor); train
        #    ONLY the RephaseHead. The base's good magnitude is locked; only phase
        #    is learned. Warm-start the base from a good magnitude checkpoint first.
        if bool(cfg.trainer.get("freeze_base", False)):
            n = 0
            for name, p in self.gen.named_parameters():
                if not name.startswith("RephaseHead"):
                    p.requires_grad = False; n += 1
            print(f"[REPHASE] froze base generator: {n} tensors frozen, "
                  f"only RephaseHead trains")

    # ------------------------------------------------------------------ #
    #  Legacy checkpoint loading (optional, via cfg.trainer.resume_path)  #
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        resume = self.cfg.trainer.get("resume_path", None)
        if resume and os.path.isfile(resume):
            print(f"[RESUME] Loading legacy checkpoint: {resume}")
            ckpt = torch.load(resume, map_location="cpu")

            # channel migration: 1ch → 2ch first conv
            gen_state = ckpt["gen"]
            conv_key  = "Denoiser.ConvReluIN_Block1.block.0.weight"
            if conv_key in gen_state:
                saved_w   = gen_state[conv_key]
                model_in  = self.gen.Denoiser.ConvReluIN_Block1.block[0].weight.shape[1]
                if saved_w.shape[1] != model_in:
                    gen_state[conv_key] = saved_w.repeat(1, model_in, 1, 1) / model_in
                    print(f"[RESUME] Migrated {conv_key}: {saved_w.shape[1]}ch → {model_in}ch")

            self.gen.load_state_dict(gen_state, strict=False)
            self.mpd.load_state_dict(ckpt["mpd"], strict=True)
            self.mrd.load_state_dict(ckpt["mrd"], strict=True)
            print("[RESUME] Model weights loaded (optimizer/scheduler not restored from legacy ckpt)")
        elif resume:
            print(f"[RESUME] Warning: {resume} not found, skipping legacy load")

        # torch.compile gives ~1.5-1.8x speedup but requires GLIBC 2.34 (Triton).
        # Skip silently on older systems (compile=false in trainer config or GLIBC too old).
        if getattr(self.cfg.trainer, "use_compile", True):
            try:
                self.gen = torch.compile(self.gen, mode="default")
                print("[COMPILE] generator compiled with torch.compile(mode='default')")
            except Exception as e:
                print(f"[COMPILE] torch.compile skipped: {e}")

    # ------------------------------------------------------------------ #
    #  Optimizers & schedulers                                            #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        t   = self.cfg.trainer
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=t.lr)
        d_params = list(self.mpd.parameters()) + list(self.mrd.parameters())
        if self.use_hf_mrd:
            d_params += list(self.mrd_hf.parameters())
        if self.use_subb_disc:
            d_params += list(self.subb_disc.parameters())
        if self.use_mel_disc:
            d_params += list(self.mel_disc.parameters())
        opt_d = torch.optim.Adam(d_params, lr=t.lr)
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=t.exp_decay)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=t.exp_decay)
        return (
            {"optimizer": opt_g, "lr_scheduler": {"scheduler": sch_g, "interval": "step"}},
            {"optimizer": opt_d, "lr_scheduler": {"scheduler": sch_d, "interval": "step"}},
        )

    @staticmethod
    def _expand_vad(vad: torch.Tensor, margin: int) -> torch.Tensor:
        """Dilate a binary (B, T) VAD mask by `margin` frames on each side."""
        if margin <= 0:
            return vad
        return F.max_pool1d(
            vad.unsqueeze(1).float(),
            kernel_size=2 * margin + 1,
            stride=1,
            padding=margin,
        ).squeeze(1)

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        x_spec, y_spec, y_wav, _x_wav, _ = batch

        # ── VAD input masking (training: use GT y_wav RMS) ─────────────
        if self.use_vad_input_masking:
            hop     = self.cfg.data.hop_length_mel
            n_fr    = y_wav.size(-1) // hop
            y_fr    = y_wav[:, 0, :n_fr * hop].view(y_wav.size(0), n_fr, hop)
            vad     = (y_fr.pow(2).mean(-1).sqrt() > self.silence_thresh).float()  # (B, T)
            vad     = self._expand_vad(vad, self.vad_margin_frames)
            x_spec  = x_spec * vad.unsqueeze(1).unsqueeze(1)   # zero silent frames

        # ── Generator forward ──────────────────────────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_hat, x_mag_pred, speech_mask = self.gen(x_spec)
        # In re-phasing mode the 3rd output is the predicted phase (B,F,T), not a mask.
        rephase_phase = speech_mask if self.use_rephase else None

        min_len = min(y_hat.shape[-1], y_wav.shape[-1])
        y_hat   = y_hat[..., :min_len]
        y_wav   = y_wav[..., :min_len]

        # ── Discriminator update ────────────────────────────────────────
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=False):
            r_mpd, g_mpd, _, _ = self.mpd(y_wav.float(), y_hat.detach().float())
            r_mrd, g_mrd, _, _ = self.mrd(y_wav.float(), y_hat.detach().float())
            r_all, g_all = list(r_mpd + r_mrd), list(g_mpd + g_mrd)
            if self.use_hf_mrd:
                r_mrd_hf, g_mrd_hf, _, _ = self.mrd_hf(y_wav.float(), y_hat.detach().float())
                r_all += list(r_mrd_hf); g_all += list(g_mrd_hf)
            if self.use_subb_disc:
                r_sb, g_sb, _, _ = self.subb_disc(y_wav.float(), y_hat.detach().float())
                r_all += [t.float() for t in r_sb]; g_all += [t.float() for t in g_sb]
            if self.use_mel_disc:
                # real = clean mel (y_spec), fake = predicted mel (denoiser output)
                r_md, g_md, _, _ = self.mel_disc(y_spec.float(), x_mag_pred.detach().float())
                r_all += [t.float() for t in r_md]; g_all += [t.float() for t in g_md]
            loss_d, _, _ = discriminator_loss(
                [t.float() for t in r_all],
                [t.float() for t in g_all])
        self.manual_backward(loss_d)
        opt_d.step()
        sch_d.step()

        # ── Generator update ────────────────────────────────────────────
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=False):
            r_mpd, g_mpd, fm_r_mpd, fm_g_mpd = self.mpd(y_wav.float(), y_hat.float())
            r_mrd, g_mrd, fm_r_mrd, fm_g_mrd = self.mrd(y_wav.float(), y_hat.float())

            loss_adv_g = (generator_loss_Ladv([t.float() for t in g_mpd])[0] +
                          generator_loss_Ladv([t.float() for t in g_mrd])[0])
            loss_fm    = (generator_loss_feature_loss(
                              [[z.float() for z in l] for l in fm_r_mpd],
                              [[z.float() for z in l] for l in fm_g_mpd]) +
                          generator_loss_feature_loss(
                              [[z.float() for z in l] for l in fm_r_mrd],
                              [[z.float() for z in l] for l in fm_g_mrd]))
            if self.use_hf_mrd:
                r_mrd_hf, g_mrd_hf, fm_r_mrd_hf, fm_g_mrd_hf = self.mrd_hf(
                    y_wav.float(), y_hat.float())
                loss_adv_g = loss_adv_g + generator_loss_Ladv(
                    [t.float() for t in g_mrd_hf])[0]
                loss_fm = loss_fm + generator_loss_feature_loss(
                    [[z.float() for z in l] for l in fm_r_mrd_hf],
                    [[z.float() for z in l] for l in fm_g_mrd_hf])
            if self.use_subb_disc:
                r_sb, g_sb, fm_r_sb, fm_g_sb = self.subb_disc(y_wav.float(), y_hat.float())
                loss_adv_g = loss_adv_g + generator_loss_Ladv(
                    [t.float() for t in g_sb])[0]
                loss_fm = loss_fm + generator_loss_feature_loss(
                    [[z.float() for z in l] for l in fm_r_sb],
                    [[z.float() for z in l] for l in fm_g_sb])
            # mel-domain adversarial sharpening of the predicted mel (own weight)
            if self.use_mel_disc:
                r_md, g_md, fm_r_md, fm_g_md = self.mel_disc(
                    y_spec.float(), x_mag_pred.float())
                loss_mel_adv = (generator_loss_Ladv([t.float() for t in g_md])[0]
                                + generator_loss_feature_loss(
                                    [[z.float() for z in l] for l in fm_r_md],
                                    [[z.float() for z in l] for l in fm_g_md]))
            else:
                loss_mel_adv = torch.zeros(1, device=self.device)

            loss_mel    = self.mel_loss_fn(y_hat.float(), y_wav.float())
            loss_mag    = self.l1(x_mag_pred.float(), y_spec.float())
            loss_mrstft = self.mrstft_loss_fn(y_hat.float(), y_wav.float())

            loss_g = (self.lam_adv * loss_adv_g
                      + self.lam_mel    * loss_mel
                      + self.lam_fm     * loss_fm
                      + self.lam_mag    * loss_mag
                      + self.lam_mrstft * loss_mrstft
                      + self.lam_mel_adv * loss_mel_adv)

            # Explicit phase supervision (real+imag STFT) — only when enabled
            if self.lam_phase > 0:
                loss_phase = self.cstft_loss_fn(y_hat.float(), y_wav.float())
                loss_g = loss_g + self.lam_phase * loss_phase
            else:
                loss_phase = torch.zeros(1, device=self.device)

            # MP-SENet anti-wrapping phase loss (re-phasing head) — only when enabled
            if self.lam_phase_aw > 0 and rephase_phase is not None:
                loss_phase_aw = self.phase_aw_loss_fn(rephase_phase.float(), y_wav.float())
                loss_g = loss_g + self.lam_phase_aw * loss_phase_aw
            else:
                loss_phase_aw = torch.zeros(1, device=self.device)

            if self.lam_mel_consist > 0:
                d  = self.cfg.data
                yh_mel = laser_mel_spec(y_hat.float().detach().clone(),
                                        d.sr, d.n_fft_mel, d.hop_length_mel,
                                        d.n_mel, d.fmax, d.max_frames)
                loss_mel_consist = self.l1(yh_mel.squeeze(1),
                                           x_mag_pred.squeeze(1).float())
                loss_g = loss_g + self.lam_mel_consist * loss_mel_consist
            else:
                loss_mel_consist = torch.zeros(1, device=self.device)

            if self.lam_energy > 0:
                hop  = self.cfg.data.hop_length_mel
                yh_f = y_hat.float().view(y_hat.size(0), -1, hop)
                y_f  = y_wav.float().view(y_wav.size(0), -1, hop)
                loss_energy = self.l1(yh_f.pow(2).mean(-1).sqrt(),
                                      y_f.pow(2).mean(-1).sqrt())
                loss_g = loss_g + self.lam_energy * loss_energy
            else:
                loss_energy = torch.zeros(1, device=self.device)

            # ── Speech mask BCE loss (new_arch only) ────────────────────
            if self.lam_mask > 0 and speech_mask is not None:
                hop       = self.cfg.data.hop_length_mel
                n_frames  = y_wav.size(-1) // hop
                y_frames  = y_wav.float()[:, 0, :n_frames * hop].view(
                                y_wav.size(0), n_frames, hop)
                y_rms     = y_frames.pow(2).mean(-1).sqrt()       # (B, T_frames)
                vad_label = (y_rms > self.silence_thresh).float() # 1=speech, 0=silence
                mask_pred = speech_mask.squeeze(2).squeeze(1)     # (B, T) from (B,1,1,T)
                # align lengths (mask is at denoiser T, vad_label at waveform T_frames)
                if mask_pred.shape[-1] != vad_label.shape[-1]:
                    mask_pred = F.interpolate(mask_pred.unsqueeze(1),
                                              size=vad_label.shape[-1],
                                              mode='linear', align_corners=False).squeeze(1)
                loss_mask = F.binary_cross_entropy(mask_pred.float(), vad_label.float())
                loss_g    = loss_g + self.lam_mask * loss_mask
            else:
                loss_mask = torch.zeros(1, device=self.device)

        self.manual_backward(loss_g)
        opt_g.step()
        sch_g.step()

        # ── Logging ────────────────────────────────────────────────────
        step = self.global_step
        if self.trainer.is_global_zero and step % 50 == 0:
            lr = opt_g.param_groups[0]["lr"]
            self.log_dict({
                "train/loss_d":           loss_d.item(),
                "train/loss_g":           loss_g.item(),
                "train/loss_adv":         loss_adv_g.item(),
                "train/loss_mel":         loss_mel.item(),
                "train/loss_fm":          loss_fm.item(),
                "train/loss_mag":         loss_mag.item(),
                "train/loss_mrstft":      loss_mrstft.item(),
                "train/loss_phase":       loss_phase.item(),
                "train/loss_phase_aw":    loss_phase_aw.item(),
                "train/loss_mel_consist": loss_mel_consist.item(),
                "train/loss_energy":      loss_energy.item(),
                "train/loss_mask":        loss_mask.item(),
                "train/lr":               lr,
            }, prog_bar=False, logger=True)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self):
        # _full   = full-waveform metrics (honest, literature-comparable)
        # _speech = speech-masked-frames metrics (legacy 0.77-style), both RMS-matched
        self._val_pesq,  self._val_stoi  = [], []          # full-waveform
        self._val_pesq_sp, self._val_stoi_sp = [], []      # speech-masked
        self._val_sisdr = []
        self._val_silence_rms = []
        self._val_mos  = {"ovrl": [], "sig": [], "bak": [], "p808": []}
        self._rec_buffers = {}   # reset each epoch; target names are persistent

    def validation_step(self, batch, batch_idx):
        x_spec, y_spec, y_wav, x_wav_raw, names = batch

        # ── GT speech mask for metric computation (all experiments) ────
        hop      = self.cfg.data.hop_length_mel
        n_fr     = y_wav.size(-1) // hop
        y_fr_val = y_wav[:, 0, :n_fr * hop].view(y_wav.size(0), n_fr, hop)
        speech_vad = (y_fr_val.pow(2).mean(-1).sqrt() > self.silence_thresh).float()
        speech_vad = self._expand_vad(speech_vad, self.vad_margin_frames)  # (B, T_frames)

        # ── VAD input masking (val/inference: laser energy proxy, no GT) ─
        if self.use_vad_input_masking:
            x_energy = x_spec.mean(dim=1).mean(dim=1)             # (B, T)
            vad      = (x_energy > self.vad_input_thresh).float()
            vad      = self._expand_vad(vad, self.vad_margin_frames)
            x_spec   = x_spec * vad.unsqueeze(1).unsqueeze(1)

        y_hat, _x_mag_pred, _mask = self.gen(x_spec)
        min_len = min(y_hat.shape[-1], y_wav.shape[-1])

        y_hat_cpu = y_hat[..., :min_len].cpu().float()
        y_wav_cpu = y_wav[..., :min_len].cpu().float()

        # Upsample speech_vad to sample level for metric masking
        speech_wav_mask = speech_vad.repeat_interleave(hop, dim=-1)  # (B, T_samples)

        sr = self.cfg.data.sr
        for j in range(x_spec.size(0)):
            ref = y_wav_cpu[j, 0].numpy()
            deg = y_hat_cpu[j, 0].numpy()

            # ── Metrics ───────────────────────────────────────────────
            mask_j    = speech_wav_mask[j, :min_len].cpu().numpy() > 0.5
            n_speech  = mask_j.sum()
            speech_fr = n_speech / max(len(mask_j), 1)

            # PESQ / STOI computed two ways, both RMS-matched (amplitude-unbiased):
            #   _full   : full continuous waveform incl. silence (honest, comparable)
            #   _speech : speech-masked frames only (legacy 0.77-style number)
            # Only compute when chunk has ≥20% voiced content.
            if speech_fr >= 0.20 and np.abs(ref).max() >= 0.01:
                ref_rms = float(np.sqrt(np.mean(ref ** 2))) + 1e-8
                deg_rms = float(np.sqrt(np.mean(deg ** 2))) + 1e-8
                deg_rms_matched = deg * (ref_rms / deg_rms)

                # ── full waveform ──
                if HAS_PESQ:
                    try: self._val_pesq.append(pesq_fn(sr, ref, deg_rms_matched, "wb"))
                    except Exception: pass
                if HAS_STOI:
                    try:
                        v = stoi_fn(ref, deg_rms_matched, sr)
                        if v > 1e-4:
                            self._val_stoi.append(v)
                    except Exception: pass

                # ── speech-masked frames (RMS-matched on speech samples) ──
                if n_speech >= sr * 0.1:
                    ref_m = ref[mask_j]
                    deg_m = deg[mask_j]
                    rm = float(np.sqrt(np.mean(ref_m ** 2))) + 1e-8
                    dm = float(np.sqrt(np.mean(deg_m ** 2))) + 1e-8
                    deg_m_matched = deg_m * (rm / dm)
                    if HAS_PESQ:
                        try: self._val_pesq_sp.append(pesq_fn(sr, ref_m, deg_m_matched, "wb"))
                        except Exception: pass
                    if HAS_STOI:
                        try:
                            v = stoi_fn(ref_m, deg_m_matched, sr)
                            if v > 1e-4:
                                self._val_stoi_sp.append(v)
                        except Exception: pass

                if HAS_MOS:
                    try:
                        m = _dnsmos_mod.run(deg.astype(np.float32), sr=sr, return_df=False)
                        for k in ("ovrl", "sig", "bak", "p808"):
                            self._val_mos[k].append(float(m[f"{k}_mos"]))
                    except Exception: pass

            # SI-SDR: scale-invariant → fine on sparse speech-only frames
            if n_speech >= sr * 0.1:
                ref_sp = ref[mask_j]; deg_sp = deg[mask_j]
                try:
                    r = ref_sp - ref_sp.mean(); e = deg_sp - deg_sp.mean()
                    s_tgt = (np.dot(e, r) / (np.dot(r, r) + 1e-9)) * r
                    self._val_sisdr.append(
                        10 * np.log10((np.dot(s_tgt, s_tgt) + 1e-9) /
                                      (np.dot(e - s_tgt, e - s_tgt) + 1e-9)))
                except Exception: pass

            # Silence RMS: how much artifact does model output on GT-silent frames?
            sil_mask = ~mask_j
            if sil_mask.sum() >= sr * 0.1:
                self._val_silence_rms.append(
                    float(np.sqrt((deg[sil_mask] ** 2).mean())))

            # ── Recording-level buffer (all chunks, incl. silent) ──────
            if self.trainer.is_global_zero:
                name = names[j]
                if (len(self._target_rec_names) < self.cfg.data.n_val_audio
                        and name not in self._target_rec_names):
                    self._target_rec_names.append(name)
                if name in self._target_rec_names:
                    if name not in self._rec_buffers:
                        self._rec_buffers[name] = {
                            "y_hat": [], "y_wav": [], "x_wav": [], "specs": []}
                    buf = self._rec_buffers[name]
                    buf["y_hat"].append(deg.copy())
                    buf["y_wav"].append(ref.copy())
                    buf["x_wav"].append(x_wav_raw[j, 0, :min_len].cpu().numpy().copy())
                    if len(buf["specs"]) < 3:
                        d = self.cfg.data
                        full_len  = d.hop_length_mel * d.max_frames
                        yh_cpu    = y_hat_cpu[j:j+1, :, :full_len]
                        yh_pad    = F.pad(yh_cpu, (0, full_len - yh_cpu.shape[-1]))
                        yhat_spec = laser_mel_spec(yh_pad, d.sr, d.n_fft_mel, d.hop_length_mel,
                                                   d.n_mel, d.fmax, d.max_frames).squeeze(0)
                        buf["specs"].append({
                            "x_spec":    x_spec[j, 0:1].cpu(),
                            "yhat_spec": yhat_spec,
                            "y_spec":    y_spec[j].cpu(),
                        })

    def on_validation_epoch_end(self):
        # ── Scalars: logged ONLY via self.log (sync_dist averages across GPUs).
        #    Do NOT also push these through wandb.log — that double-logs the key
        #    and the rank-0-local value can overwrite the synced one. ──────────
        def _mean(x): return float(np.mean(x)) if x else 0.0
        pesq_full   = _mean(self._val_pesq)
        stoi_full   = _mean(self._val_stoi)
        pesq_speech = _mean(self._val_pesq_sp)
        stoi_speech = _mean(self._val_stoi_sp)
        sisdr_local = _mean(self._val_sisdr)
        sil_rms     = _mean(self._val_silence_rms)

        # Checkpoint monitor key. Keep `val/pesq` == full-waveform PESQ.
        self.log("val/pesq",        pesq_full,   sync_dist=True, on_epoch=True)
        self.log("val/stoi",        stoi_full,   sync_dist=True, on_epoch=True)  # == stoi_full
        self.log("val/pesq_full",   pesq_full,   sync_dist=True, on_epoch=True)
        self.log("val/stoi_full",   stoi_full,   sync_dist=True, on_epoch=True)
        self.log("val/pesq_speech", pesq_speech, sync_dist=True, on_epoch=True)
        self.log("val/stoi_speech", stoi_speech, sync_dist=True, on_epoch=True)
        self.log("val/si_sdr",      sisdr_local, sync_dist=True, on_epoch=True)
        self.log("val/silence_rms", sil_rms,     sync_dist=True, on_epoch=True)
        for k, v in self._val_mos.items():
            if v: self.log(f"val/mos_{k}", float(np.mean(v)), sync_dist=True, on_epoch=True)

        if not self.trainer.is_global_zero:
            return

        # ── Greppable stdout line so phase progress (SI-SDR) is visible in the
        #    slurm .out without opening WandB:  grep '\[VAL\]' logs/.../*.out
        #    Prefer the sync_dist-averaged values from callback_metrics; fall back
        #    to the rank-0-local means if they aren't populated yet. ──────────────
        def _cm(key, local):
            v = self.trainer.callback_metrics.get(key, None)
            return float(v) if v is not None else local
        print(
            f"[VAL] epoch={self.current_epoch} "
            f"si_sdr={_cm('val/si_sdr', sisdr_local):+.2f} "
            f"pesq={_cm('val/pesq_full', pesq_full):.3f} "
            f"stoi={_cm('val/stoi_full', stoi_full):.3f} "
            f"pesq_sp={_cm('val/pesq_speech', pesq_speech):.3f} "
            f"stoi_sp={_cm('val/stoi_speech', stoi_speech):.3f}",
            flush=True)
        sr  = self.cfg.data.sr
        # `log` now carries ONLY media (audio + spectrograms), never scalars.
        log = {}

        for rec_name in self._target_rec_names:
            buf = self._rec_buffers.get(rec_name)
            if not buf or not buf["y_hat"]:
                continue

            # ── First 3 consecutive chunks: spec panel + audio triple ──
            for ci, spec_data in enumerate(buf["specs"]):
                fig, axes = plt.subplots(1, 3, figsize=(15, 3))
                for ax, spec, title in zip(
                        axes,
                        [spec_data["x_spec"], spec_data["yhat_spec"], spec_data["y_spec"]],
                        ["Laser input", "Reconstructed", "Clean target"]):
                    ax.imshow(spec.squeeze(0).numpy(), origin="lower", aspect="auto", cmap="magma")
                    ax.set_title(title, fontsize=9); ax.axis("off")
                plt.suptitle(f"{rec_name}  chunk-{ci}", fontsize=7); plt.tight_layout()
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                img = np.frombuffer(fig.canvas.buffer_rgba(),
                                    dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
                log[f"val/spec/{rec_name}/chunk{ci}"] = wandb.Image(img)
                plt.close(fig)
                log[f"val/chunks/{rec_name}/{ci}/laser"] = wandb.Audio(
                    buf["x_wav"][ci], sample_rate=sr,
                    caption=f"{rec_name} chunk{ci} — laser input")
                log[f"val/chunks/{rec_name}/{ci}/recon"] = wandb.Audio(
                    buf["y_hat"][ci], sample_rate=sr,
                    caption=f"{rec_name} chunk{ci} — reconstructed")
                log[f"val/chunks/{rec_name}/{ci}/clean"] = wandb.Audio(
                    buf["y_wav"][ci], sample_rate=sr,
                    caption=f"{rec_name} chunk{ci} — clean target")

            # ── Full stitched recording ────────────────────────────────
            y_hat_full = np.concatenate(buf["y_hat"])
            y_wav_full = np.concatenate(buf["y_wav"])
            x_wav_full = np.concatenate(buf["x_wav"])
            log[f"val/full/{rec_name}/laser"] = wandb.Audio(
                x_wav_full, sample_rate=sr,
                caption=f"{rec_name} — full laser input")
            log[f"val/full/{rec_name}/recon"] = wandb.Audio(
                y_hat_full, sample_rate=sr,
                caption=f"{rec_name} — full reconstructed")
            log[f"val/full/{rec_name}/clean"] = wandb.Audio(
                y_wav_full, sample_rate=sr,
                caption=f"{rec_name} — full clean target")

        wandb.log(log, step=self.global_step)
