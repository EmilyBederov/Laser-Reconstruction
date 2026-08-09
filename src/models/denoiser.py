import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from .conformer import ConformerBlock

Ce = 64


class ConvReluIN(nn.Module):
    def __init__(self, Cin=1, Ce=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(Cin, Ce, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.InstanceNorm2d(Ce, affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class DilConvInRelu4TIMES(nn.Module):
    def __init__(self, Ce=64):
        super().__init__()
        self.DilBlock1 = nn.Sequential(
            nn.Conv2d(Ce, Ce, kernel_size=(3,3), dilation=(1,1), padding=(1,1), bias=True),
            nn.InstanceNorm2d(Ce, affine=True), nn.ReLU(inplace=True))
        self.DilBlock2 = nn.Sequential(
            nn.Conv2d(Ce, Ce, kernel_size=(3,3), dilation=(2,2), padding=(2,2), bias=True),
            nn.InstanceNorm2d(Ce, affine=True), nn.ReLU(inplace=True))
        self.DilBlock3 = nn.Sequential(
            nn.Conv2d(Ce, Ce, kernel_size=(3,3), dilation=(4,4), padding=(4,4), bias=True),
            nn.InstanceNorm2d(Ce, affine=True), nn.ReLU(inplace=True))
        self.DilBlock4 = nn.Sequential(
            nn.Conv2d(Ce, Ce, kernel_size=(3,3), dilation=(8,8), padding=(8,8), bias=True),
            nn.InstanceNorm2d(Ce, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.DilBlock1(x) + x
        x2 = self.DilBlock2(x1) + x1
        x3 = self.DilBlock3(x2) + x2
        return self.DilBlock4(x3) + x3


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super().__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel, dim_head=num_channel // 4, heads=4,
            conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(
            dim=num_channel, dim_head=num_channel // 4, heads=4,
            conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        assert f <= 512, f"TSCB: F must be <= 512, got F={f}"
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        return x_f.view(b, t, f, c).permute(0, 3, 1, 2)


class DenoiseClass(nn.Module):
    # INPUT:  (B, 2, F, T)  — two laser channels
    # OUTPUT: (B, 1, F, T), feat0 (B, Ce, F, T)
    def __init__(self, extra_tscb: int = 0):
        super().__init__()
        self.ConvReluIN_Block1         = ConvReluIN(Cin=2)
        self.DilConvInRelu4TIMES_Block1 = DilConvInRelu4TIMES()
        self.TSCB_Block1               = TSCB()
        self.TSCB_Block2               = TSCB()
        self.TSCB_Block3               = TSCB()
        self.TSCB_Block4               = TSCB()
        # Optional extra TSCB capacity (default 0 = unchanged). More depth on the
        # magnitude producer for the mel-sharpening experiment.
        self.extra_tscb = nn.ModuleList([TSCB() for _ in range(extra_tscb)])
        self.DilConvInRelu4TIMES_Block2 = DilConvInRelu4TIMES()
        self.Last2DConvLayer = nn.Conv2d(Ce, 1, kernel_size=(3,3), padding=(1,1))

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == 2, f"Expected (B,2,F,T), got {tuple(x.shape)}"
        feat0 = self.ConvReluIN_Block1(x)
        feat1 = self.DilConvInRelu4TIMES_Block1(feat0)
        y = feat1.permute(0, 1, 3, 2)
        y = ckpt.checkpoint(self.TSCB_Block1, y, use_reentrant=False)
        y = ckpt.checkpoint(self.TSCB_Block2, y, use_reentrant=False)
        y = ckpt.checkpoint(self.TSCB_Block3, y, use_reentrant=False)
        y = ckpt.checkpoint(self.TSCB_Block4, y, use_reentrant=False)
        for blk in self.extra_tscb:
            y = ckpt.checkpoint(blk, y, use_reentrant=False)
        y = y.permute(0, 1, 3, 2)
        y = self.DilConvInRelu4TIMES_Block2(y + feat1)
        y = self.Last2DConvLayer(y + feat0)
        return y, feat0
