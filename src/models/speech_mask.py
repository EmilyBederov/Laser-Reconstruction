import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResLayer(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation,
                      padding=dilation, groups=ch),   # depthwise
            nn.Conv1d(ch, ch, kernel_size=1),          # pointwise
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class SpeechMaskPredictor(nn.Module):
    """
    Predicts a per-frame soft speech/silence mask from denoiser feat0.

    Deeper dilations (1,2,4,8,16,32,64) give ~500-frame receptive field
    (~4 seconds at hop=128, sr=16000) so the model can see well before
    a speech onset and suppress laser noise confidently.

    No threshold at inference — threshold is only used during training
    to build VAD pseudo-labels from the clean reference.

    Input:  feat0 (B, Ce=64, F, T)
    Output: mask  (B, 1,     1, T)  — values in [0,1], 1=speech, 0=silence
    """

    DILATIONS = (1, 2, 4, 8, 16, 32, 64)

    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, ch, F, T) → (B, ch, 1, T)

        self.temporal = nn.Sequential(
            *[DilatedResLayer(in_ch, d) for d in self.DILATIONS]
        )

        self.out = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat0: torch.Tensor) -> torch.Tensor:
        # feat0: (B, Ce, F, T)
        x = self.freq_pool(feat0).squeeze(2)   # (B, Ce, T)
        x = self.temporal(x)                   # (B, Ce, T)
        mask = self.out(x)                     # (B, 1,  T)
        return mask.unsqueeze(2)               # (B, 1,  1, T) — broadcastable over F
