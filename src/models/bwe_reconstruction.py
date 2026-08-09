import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .speech_mask import SpeechMaskPredictor
from .snake import Snake

C = 512
k = 2


class Conv1DPrev(nn.Module):
    def __init__(self, FreqDim):
        super().__init__()
        self.freqdim = FreqDim
        self.Block = nn.Sequential(
            nn.Conv1d(FreqDim, C, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(C), nn.ReLU())

    def forward(self, x):  # (B, 1, F, T)
        assert x.size(2) == self.freqdim
        return self.Block(x.squeeze(1))  # (B, C, T)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch*k*2, 1), nn.BatchNorm1d(in_ch*k*2), nn.GLU(dim=1))
        self.s2 = nn.Sequential(
            nn.Conv1d(in_ch*k, in_ch*k*2, 9, padding=4, groups=in_ch*k),
            nn.BatchNorm1d(in_ch*k*2), nn.GLU(dim=1))
        self.s3 = nn.Sequential(
            nn.Conv1d(in_ch*k, in_ch*k*2, 1), nn.BatchNorm1d(in_ch*k*2), nn.GLU(dim=1))
        self.skip = nn.Conv1d(in_ch, in_ch*k, 1)
        self.proj = nn.Conv1d(in_ch*k, in_ch, 1)
        self.act  = nn.ReLU()

    def forward(self, x):
        y = self.s3(self.s2(self.s1(x)) + self.skip(x))
        return self.act(self.proj(y))


class ConvFlashAttn(nn.Module):
    """Paper-original ConvFlash: FFNN → FlashAttn → ResConvBlock → FFNN → LayerNorm.
    Uses nn.MultiheadAttention (PyTorch 2.x dispatches to Flash kernels on BF16/FP16).
    head_dim fixed at 32 across all channel widths."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        num_heads = max(1, channels // 32)
        self.ffnn1 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.attn  = nn.MultiheadAttention(channels, num_heads,
                                            dropout=dropout, batch_first=True)
        self.res   = ResConvBlock(channels)
        self.ffnn2 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.ln    = nn.LayerNorm(channels)

    def forward(self, x):  # (B, C, T)
        x = x + self.ffnn1(x)
        y = x.transpose(1, 2)                              # (B, T, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y.transpose(1, 2)                         # (B, C, T)
        x = self.res(x)
        x = x + self.ffnn2(x)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _SEGate(nn.Module):
    """Squeeze-and-Excitation: global avg pool → FC → sigmoid channel gate. O(C²/r)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.SiLU(),
            nn.Linear(mid, channels), nn.Sigmoid())

    def forward(self, x):  # (B, C, T)
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class ConvFlashWindowedAttnSE(nn.Module):
    """Exp 9: FFNN → windowed MHA (O(W²), all 4 blocks) → ResConv → SE gate → FFNN → LN.
    window_size=512 ≈ 4s at hop=128. Replaces O(T²) full attention with local windows."""
    def __init__(self, channels, window_size=512, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        num_heads = max(1, channels // 32)
        self.ffnn1 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.attn  = nn.MultiheadAttention(channels, num_heads,
                                            dropout=dropout, batch_first=True)
        self.res   = ResConvBlock(channels)
        self.se    = _SEGate(channels)
        self.ffnn2 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.ln    = nn.LayerNorm(channels)

    def forward(self, x):  # (B, C, T)
        x = x + self.ffnn1(x)
        B, C, T = x.shape
        W   = min(self.window_size, T)
        pad = (-T) % W
        xp  = F.pad(x, (0, pad)) if pad else x         # (B, C, T_pad)
        n_w = xp.shape[-1] // W
        xw  = xp.transpose(1, 2).reshape(B * n_w, W, C)
        yw, _ = self.attn(xw, xw, xw, need_weights=False)
        yw  = yw.reshape(B, -1, C).transpose(1, 2)[:, :, :T]  # (B, C, T)
        x   = x + yw
        x   = self.res(x)
        x   = self.se(x)
        x   = x + self.ffnn2(x)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ConvFlashLinearAttnSE(nn.Module):
    """Exp 10: FFNN → ReLU linear attention (O(T·d²), full context) → ResConv → SE → FFNN → LN.
    No T² cost → safe for all 4 blocks. Full sequence context via kernel trick."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        num_heads      = max(1, channels // 32)
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.ffnn1 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.qkv   = nn.Linear(channels, channels * 3, bias=False)
        self.out   = nn.Linear(channels, channels, bias=False)
        self.res   = ResConvBlock(channels)
        self.se    = _SEGate(channels)
        self.ffnn2 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.ln    = nn.LayerNorm(channels)

    def forward(self, x):  # (B, C, T)
        x = x + self.ffnn1(x)
        B, C, T = x.shape
        h, d = self.num_heads, self.head_dim
        xt = x.transpose(1, 2)                                  # (B, T, C)
        qkv = self.qkv(xt).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)                                 # each (B, h, T, d)
        Q, K = F.relu(Q), F.relu(K)                             # non-negative feature map
        KV  = torch.einsum('bhtd,bhte->bhde', K, V)             # (B, h, d, d)
        Z   = torch.einsum('bhtd,bhd->bht', Q, K.sum(2))        # (B, h, T) normaliser
        y   = torch.einsum('bhtd,bhde->bhte', Q, KV)            # (B, h, T, d)
        y   = y / (Z.unsqueeze(-1) + 1e-6)
        y   = self.out(y.transpose(1, 2).reshape(B, T, C))      # (B, T, C)
        x   = x + y.transpose(1, 2)                             # (B, C, T)
        x   = self.res(x)
        x   = self.se(x)
        x   = x + self.ffnn2(x)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ConvFlashFF(nn.Module):
    """Improved dilated path: FFNN → wider ConvFlash → ResConvBlock → FFNN → LN.
    Mirrors ConvFlashAttn structure but replaces O(T²) MHA with O(T·k) dilated conv.
    Safe for all 4 upsample blocks including long T (blocks 3 and 4)."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.ffnn1 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.conv = ConvFlash(channels, dilations=(1, 2, 4, 8, 16, 32))
        self.res  = ResConvBlock(channels)
        self.ffnn2 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1))
        self.ln   = nn.LayerNorm(channels)

    def forward(self, x):  # (B, C, T)
        x = x + self.ffnn1(x)
        x = self.conv(x)          # ConvFlash has its own residual connection
        x = self.res(x)
        x = x + self.ffnn2(x)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ConvFlash(nn.Module):
    def __init__(self, channels, dilations=(1,2,4,8), kernel_size=9, num_groups=8, dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.Conv1d(channels, channels, 1), nn.SiLU())
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=(kernel_size//2)*d, groups=channels),
                nn.GroupNorm(num_groups, channels), nn.SiLU())
            for d in dilations])
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(dilations), channels, 1), nn.Dropout(dropout))
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.pre(x)
        y = self.fuse(torch.cat([b(y) for b in self.branches], dim=1))
        return self.out_act(x + y)


class TransposeConvBNReLU(nn.Module):
    """Original upsample block. Unchanged so existing checkpoints still load."""
    def __init__(self, in_ch, upsample_factor):
        super().__init__()
        s = upsample_factor
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, in_ch//2, 2*s, stride=s,
                               padding=s//2, output_padding=s%2),
            nn.BatchNorm1d(in_ch//2), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class TransposeConvBNSnake(nn.Module):
    """Same shape as TransposeConvBNReLU, but with Snake activation (BigVGAN-style)
    instead of ReLU. New class: only instantiated when use_snake=True so old
    checkpoints (which never saw this class) are unaffected."""
    def __init__(self, in_ch, upsample_factor):
        super().__init__()
        s = upsample_factor
        out_ch = in_ch // 2
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, 2*s, stride=s,
                                         padding=s//2, output_padding=s%2)
        self.norm   = nn.BatchNorm1d(out_ch)
        self.snake  = Snake(out_ch)

    def forward(self, x):
        return self.snake(self.norm(self.deconv(x)))


class UpSampleClass(nn.Module):
    def __init__(self, in_ch, factor, use_snake: bool = False,
                 use_attn_recon: bool = False, use_ffn_dil: bool = False,
                 use_windowed_attn_se: bool = False, use_linear_attn_se: bool = False,
                 use_wide_dil: bool = False, recon_extra_flash: int = 0):
        super().__init__()
        self.use_attn_recon      = use_attn_recon
        self.use_ffn_dil         = use_ffn_dil
        self.use_windowed_attn_se = use_windowed_attn_se
        self.use_linear_attn_se  = use_linear_attn_se
        self.up = (TransposeConvBNSnake(in_ch, factor) if use_snake
                   else TransposeConvBNReLU(in_ch, factor))
        ch = in_ch // 2

        def _make_flash():
            if use_attn_recon:
                return ConvFlashAttn(ch)
            if use_ffn_dil:
                return ConvFlashFF(ch)
            if use_windowed_attn_se:
                return ConvFlashWindowedAttnSE(ch)
            if use_linear_attn_se:
                return ConvFlashLinearAttnSE(ch)
            # basic block; use_wide_dil widens the receptive field (1-8 → 1-16)
            # while keeping the lean ResConv+ConvFlash structure (no FFNN/LN).
            return ConvFlash(ch, dilations=(1, 2, 4, 8, 16) if use_wide_dil
                                 else (1, 2, 4, 8))

        if not (use_attn_recon or use_ffn_dil or use_windowed_attn_se or use_linear_attn_se):
            self.res = ResConvBlock(ch)
        self.flash = _make_flash()
        # Extra reconstructor depth (default 0 = unchanged, so existing checkpoints
        # load: no new params are created unless recon_extra_flash > 0).
        self.extra_flash = nn.ModuleList([_make_flash() for _ in range(recon_extra_flash)])

    @property
    def _has_inner_res(self):
        return (self.use_attn_recon or self.use_ffn_dil or
                self.use_windowed_attn_se or self.use_linear_attn_se)

    def forward(self, x):
        x = self.up(x)
        if not self._has_inner_res:
            x = self.res(x)
        x = self.flash(x)
        for blk in self.extra_flash:
            x = blk(x)
        return x


class ReconstrucClass(nn.Module):
    _SKIP_CH = [C, C//2, C//4, C//8, C//16]
    _SKIP_UP = [1,  4,    16,   64,   128]

    def __init__(self, FreqDim, use_sinusoidal_pe=False,
                 use_multi_scale_skips=True, use_speech_mask=False,
                 use_snake: bool = False, use_attn_recon: bool = False,
                 use_ffn_dil: bool = False,
                 use_windowed_attn_se: bool = False,
                 use_linear_attn_se: bool = False,
                 use_istft_head: bool = False,
                 use_wide_dil: bool = False,
                 recon_extra_flash: int = 0):
        super().__init__()
        self.use_sinusoidal_pe     = use_sinusoidal_pe
        self.use_multi_scale_skips = use_multi_scale_skips
        self.use_speech_mask       = use_speech_mask
        self.use_snake             = use_snake
        self.use_attn_recon        = use_attn_recon
        self.use_ffn_dil           = use_ffn_dil
        self.use_windowed_attn_se  = use_windowed_attn_se
        self.use_linear_attn_se    = use_linear_attn_se
        self.use_istft_head        = use_istft_head
        self.use_wide_dil          = use_wide_dil

        self.Conv1DPrev_Block = Conv1DPrev(FreqDim)

        # ── Vocos-style iSTFT head: explicit phase. Predicts the complex STFT at
        #    frame rate (no time-domain upsampling), then ONE inverse STFT.
        #    n_fft=1024, hop=128 → 768 frames reconstruct 98304 samples (×128). ──
        if use_istft_head:
            self.istft_n_fft = 1024
            self.istft_hop   = 128
            n_bins = self.istft_n_fft // 2 + 1          # 513 complex bins
            self.istft_body = nn.Sequential(
                *[ConvFlash(C, dilations=(1, 2, 4, 8, 16, 32)) for _ in range(6)])
            self.istft_head = nn.Conv1d(C, 2 * n_bins, kernel_size=1)  # → real | imag
        # Attention: blocks 1-2 only (O(T²) OOMs at blocks 3-4).
        # All other variants (ConvFlashFF, windowed, linear): safe for all 4 blocks.
        def _up(in_ch, factor):
            return UpSampleClass(in_ch, factor, use_snake=use_snake,
                                 use_attn_recon=use_attn_recon and (in_ch >= C // 2),
                                 use_ffn_dil=use_ffn_dil,
                                 use_windowed_attn_se=use_windowed_attn_se,
                                 use_linear_attn_se=use_linear_attn_se,
                                 use_wide_dil=use_wide_dil,
                                 recon_extra_flash=recon_extra_flash)
        self.UpSampleClassBlock1_Block = _up(C,     4)
        self.UpSampleClassBlock2_Block = _up(C//2,  4)
        self.UpSampleClassBlock3_Block = _up(C//4,  4)
        self.UpSampleClassBlock4_Block = _up(C//8,  2)
        self.Conv1DLast = nn.Conv1d(C//16, 1, 7, padding=3)

        if use_multi_scale_skips:
            self.skip_projs = nn.ModuleList(
                [nn.Conv1d(64, ch, 1) for ch in self._SKIP_CH])
            self.skip_alphas = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in self._SKIP_CH])

        if use_sinusoidal_pe:
            self.pe_alpha = nn.Parameter(torch.zeros(1))

        if use_speech_mask:
            self.mask_predictor = SpeechMaskPredictor(in_ch=64)

    @staticmethod
    def _sinusoidal_pe(T, d, device):
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
        dim = torch.arange(0, d, 2, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.pow(10000.0, dim / d)
        pe  = torch.zeros(1, d, T, device=device)
        pe[0, 0::2] = torch.sin(pos / div)
        pe[0, 1::2] = torch.cos(pos / div)
        return pe

    def _skip(self, s, stage):
        up = self._SKIP_UP[stage]
        if up > 1:
            s = F.interpolate(s, scale_factor=up, mode='nearest')
        return self.skip_alphas[stage] * self.skip_projs[stage](s)

    def forward(self, x, feat0):  # x=(B,1,F,T), feat0=(B,64,F,T)
        s = feat0.mean(dim=2) if self.use_multi_scale_skips else None

        # predict speech mask early from feat0 (before upsampling)
        # mask: (B, 1, 1, T) — values in [0,1], 1=speech 0=silence
        mask = self.mask_predictor(feat0) if self.use_speech_mask else None

        x = self.Conv1DPrev_Block(x)                           # (B, C, T)

        if self.use_sinusoidal_pe:
            x = x + self.pe_alpha * self._sinusoidal_pe(x.size(-1), C, x.device)

        # ── Vocos-style iSTFT head: predict complex STFT at frame rate → iSTFT ──
        if self.use_istft_head:
            if self.use_multi_scale_skips:
                x = x + self._skip(s, 0)                       # frame-rate skip only
            x = self.istft_body(x)                             # (B, C, T)
            ri = self.istft_head(x)                            # (B, 2*n_bins, T)
            n_bins = self.istft_n_fft // 2 + 1
            real = ri[:, :n_bins].float()
            imag = ri[:, n_bins:].float()
            spec = torch.complex(real, imag)                   # (B, n_bins, T)
            win = torch.hann_window(self.istft_n_fft, device=x.device)
            L = self.istft_hop * x.shape[-1]                   # 128 × T samples
            y_hat = torch.istft(spec, self.istft_n_fft, self.istft_hop,
                                self.istft_n_fft, win, center=True, length=L)
            return y_hat.unsqueeze(1), mask                    # (B, 1, 128T)

        if self.use_multi_scale_skips:
            x = x + self._skip(s, 0)

        x = self.UpSampleClassBlock1_Block(x)
        if self.use_multi_scale_skips:
            x = x + self._skip(s, 1)

        x = self.UpSampleClassBlock2_Block(x)
        if self.use_multi_scale_skips:
            x = x + self._skip(s, 2)

        x = self.UpSampleClassBlock3_Block(x)
        if self.use_multi_scale_skips:
            x = x + self._skip(s, 3)

        x = self.UpSampleClassBlock4_Block(x)
        if self.use_multi_scale_skips:
            x = x + self._skip(s, 4)

        y_hat = self.Conv1DLast(x)                             # (B, 1, 128T)

        if mask is not None:
            # upsample mask from T → 128T and gate the waveform
            gate = F.interpolate(mask.squeeze(2), size=y_hat.shape[-1], mode='linear',
                                 align_corners=False)          # (B, 1, 128T)
            y_hat = y_hat * gate

        return y_hat, mask                                     # mask is None if not used
