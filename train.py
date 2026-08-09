"""AudioLaser training entry point.

    python train.py                          # trains the default model (audiolaser)
    python train.py model=base_attn trainer=base_attn
    python train.py model=phase_riloss trainer=phase_riloss   # RI-loss phase variant

Checkpoints are written to <trainer.ckpt_dir> (relative to the run dir). Training
auto-resumes from <ckpt_dir>/last.ckpt if present. Set WANDB_MODE=disabled to run
without Weights & Biases logging.
"""
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.io import TorchCheckpointIO


class _LegacyCheckpointIO(TorchCheckpointIO):
    """Load checkpoints with weights_only=False and strip torch.compile prefixes.
    The generator is wrapped in torch.compile(), so saved checkpoints carry a
    'gen._orig_mod.' prefix; strip it so a strict load works before compilation."""
    def load_checkpoint(self, path, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        ckpt = super().load_checkpoint(path, map_location=map_location, **kwargs)
        sd = ckpt.get("state_dict")
        if sd and any("_orig_mod." in k for k in sd):
            ckpt["state_dict"] = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        return ckpt


from src.lightning.system import AudioLaserSystem
from src.data.dataset import LaserDataModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dm     = LaserDataModule(cfg)
    system = AudioLaserSystem(cfg)

    # checkpoints: <trainer.ckpt_dir> (relative to the Hydra run dir), one dir per model
    ckpt_dir = cfg.trainer.get("ckpt_dir", f"checkpoints/{cfg.model.name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath                 = ckpt_dir,
        filename                = "epoch{epoch:03d}-pesq{val/pesq:.3f}",
        auto_insert_metric_name = False,
        monitor                 = "val/pesq",
        mode                    = "max",
        save_top_k              = 3,
        save_last               = True,
        every_n_epochs          = cfg.data.val_every_n_epochs,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=50)

    # logger: Weights & Biases if available/enabled, else a local CSV logger
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        logger = CSVLogger(save_dir=".", name="logs")
    else:
        logger = WandbLogger(
            project = cfg.data.wandb_project,
            name    = cfg.trainer.get("wandb_run_name", cfg.model.name),
            config  = OmegaConf.to_container(cfg, resolve=True),
        )

    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    trainer = pl.Trainer(
        max_epochs        = cfg.trainer.epochs,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        devices           = n_gpus if torch.cuda.is_available() else 1,
        strategy          = DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        precision         = "bf16-mixed",
        plugins           = [_LegacyCheckpointIO()],
        callbacks         = [checkpoint_cb, lr_monitor, progress_bar],
        logger            = logger,
        log_every_n_steps = 50,
        check_val_every_n_epoch = cfg.data.val_every_n_epochs,
    )

    # auto-resume from last.ckpt if present (full optimizer/scheduler state)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.isfile(last_ckpt) else None
    if ckpt_path:
        print(f"[RESUME] Resuming from {ckpt_path}")
    trainer.fit(system, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
