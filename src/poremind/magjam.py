"""MAGJAM models for waveform-based event classification.

This module is intentionally self-contained.  The PoreMind workflow imports it
lazily only when ``model_name`` is ``MAGJAM`` or ``MAGJAM_d4`` so that the
existing API remains importable in environments without PyTorch.

The public model names are:

``MAGJAM``
    The depth-8 MSS + Jamba-style hybrid model.
``MAGJAM_d4``
    The depth-4 variant.

Inputs use the shape ``[batch, 1, time]``.  The classifier head is kept
separate from :class:`MAGJAMExtractor` because PoreMind's DL workflow stores
models as ``ModuleDict({'extractor': ..., 'head': ...})`` and reuses that
layout for prediction and Integrated Gradients attribution.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_at_T(src: torch.Tensor, T: int) -> torch.Tensor:
    """Return a detached magnitude mask with temporal length ``T``."""
    if src.dim() == 4:
        src = src.squeeze(1)
    if src.dim() == 2:
        src = src.unsqueeze(1)
    if src.dim() != 3:
        raise ValueError(
            "_mask_at_T expects a 2D/3D/compatible 4D tensor, "
            f"got shape={tuple(src.shape)}"
        )

    weight = src.detach().abs().mean(dim=1, keepdim=True)
    if weight.shape[-1] != T:
        weight = F.adaptive_avg_pool1d(weight, T)
    return weight


class MagnitudePool1d(nn.Module):
    """Magnitude-weighted temporal pooling."""

    def forward(self, x: torch.Tensor, raw_x: torch.Tensor | None = None) -> torch.Tensor:
        source = raw_x if raw_x is not None else x
        weight = _mask_at_T(source, x.shape[-1])
        weight = weight / (weight.sum(dim=2, keepdim=True) + 1e-6)
        return (x * weight).sum(dim=2)


class ConcatPool1d(nn.Module):
    """Average/max pooling followed by a 1x1 convolutional fusion."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Conv1d(in_c * 2, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor, raw_x: torch.Tensor | None = None) -> torch.Tensor:
        del raw_x
        avg = x.mean(dim=2)
        maximum = x.max(dim=2).values
        h = torch.cat([avg, maximum], dim=1)
        return self.proj(h.unsqueeze(2)).squeeze(2)


class MagGAP(nn.Module):
    """Magnitude-weighted pooling plus residual global average pooling."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.ln = nn.LayerNorm(in_c)

    def forward(self, x: torch.Tensor, raw_x: torch.Tensor | None = None) -> torch.Tensor:
        source = raw_x if raw_x is not None else x
        weight = _mask_at_T(source, x.shape[-1])
        weight = weight / (weight.sum(dim=2, keepdim=True) + 1e-6)
        magnitude = (x * weight).sum(dim=2)
        gap = x.mean(dim=2)
        return self.ln(magnitude + gap)


class FastSSMBlock(nn.Module):
    """Lightweight Mamba-style gated depthwise temporal block."""

    def __init__(self, dim: int, kernel: int = 7):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FastSSMBlock expects [B, T, C], got shape={tuple(x.shape)}")
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_main = self.dw_conv(x_main.transpose(1, 2)).transpose(1, 2)
        z = torch.sigmoid(self.gate(z))
        return self.out_proj(x_main * z)


class Mamba1DFast(nn.Module):
    """Standalone simplified Mamba-style 1D classifier."""

    def __init__(
        self,
        n_classes: int,
        T: int,
        dim: int = 96,
        depth: int = 4,
        kernel: int = 7,
    ):
        super().__init__()
        self.T = T
        self.dim = dim
        self.depth = depth
        self.stem = nn.Sequential(nn.Conv1d(1, dim, kernel_size=7, padding=3))
        self.blocks = nn.ModuleList([FastSSMBlock(dim, kernel=kernel) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x).transpose(1, 2)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class SE1D(nn.Module):
    """Squeeze-and-excitation channel attention for ``[B, T, C]`` tensors."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=1)
        z = F.gelu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    """Pre-LN SwiGLU feed-forward residual block."""

    def __init__(self, dim: int, mult: int = 2, init: float = 1e-4):
        super().__init__()
        hidden = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.scale = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.w1(h)) * self.w2(h)
        return x + self.w3(h) * self.scale


class MSSHybridBlock(nn.Module):
    """FastSSM + SE + SwiGLU hybrid residual block."""

    def __init__(
        self,
        dim: int,
        kernel: int = 7,
        mult: int = 2,
        reduction: int = 4,
        drop_path: float = 0.0,
        init: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = FastSSMBlock(dim, kernel=kernel)
        self.ls1 = nn.Parameter(init * torch.ones(dim))
        self.se = SE1D(dim, reduction=reduction)
        self.ffn = SwiGLUFFN(dim, mult=mult, init=init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mamba(self.norm1(x)) * self.ls1
        x = x + self.drop_path(mixed)
        x = self.se(x)
        return self.ffn(x)


class MSSJambaHybrid(nn.Module):
    """MSS + Jamba-style hybrid classifier used by MAGJAM."""

    def __init__(
        self,
        n_classes: int,
        T: int,
        depth: int = 6,
        dim: int = 128,
        kernel: int = 7,
        mult: int = 2,
        reduction: int = 4,
        drop_path: float = 0.1,
        max_ls: float = 1e-4,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.T = T
        self.depth = depth
        self.dim = dim

        self.stem = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                MSSHybridBlock(
                    dim=dim,
                    kernel=kernel,
                    mult=mult,
                    reduction=reduction,
                    drop_path=dpr[i],
                    init=max_ls,
                )
                for i in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.pools = nn.ModuleList([MagGAP(dim, dim) for _ in range(depth)])
        self.feat_dim = depth * dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the multi-stage feature vector before classification."""
        raw = x
        h = self.stem(x).transpose(1, 2)
        features = []
        for block, norm, pool in zip(self.blocks, self.norms, self.pools):
            h = block(h)
            h_norm = norm(h).transpose(1, 2)
            features.append(pool(h_norm, raw_x=raw))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class MAGJAMExtractor(nn.Module):
    """PoreMind-compatible MAGJAM feature extractor.

    Only the backbone modules are registered here.  The classifier head is
    registered separately by ``build_DL_model`` so that the resulting
    ``ModuleDict`` has the same structure as the existing DL models without
    registering the MAGJAM head twice.
    """

    def __init__(self, model: MSSJambaHybrid):
        super().__init__()
        self.stem = model.stem
        self.blocks = model.blocks
        self.norms = model.norms
        self.pools = model.pools
        self.depth = model.depth
        self.dim = model.dim
        self.feat_dim = model.feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x
        h = self.stem(x).transpose(1, 2)
        features = []
        for block, norm, pool in zip(self.blocks, self.norms, self.pools):
            h = block(h)
            h_norm = norm(h).transpose(1, 2)
            features.append(pool(h_norm, raw_x=raw))
        return torch.cat(features, dim=-1)


class MAGJAMFullModel(nn.Module):
    """Standalone MAGJAM classifier wrapper for direct PyTorch use."""

    def __init__(self, n_classes: int, T: int, depth: int = 8):
        super().__init__()
        backbone = MSSJambaHybrid(n_classes=n_classes, T=T, depth=depth)
        self.extractor = MAGJAMExtractor(backbone)
        self.head = backbone.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extractor(x))


def mss_jamba_hybrid(n_classes: int = 15, T: int = 1000) -> MSSJambaHybrid:
    return MSSJambaHybrid(n_classes=n_classes, T=T, depth=6, dim=128, kernel=7, mult=2)


def mss_jamba_hybrid_d4(n_classes: int = 15, T: int = 1000) -> MSSJambaHybrid:
    return MSSJambaHybrid(n_classes=n_classes, T=T, depth=4, dim=128, kernel=7, mult=2)


def mss_jamba_hybrid_d8(n_classes: int = 15, T: int = 1000) -> MSSJambaHybrid:
    return MSSJambaHybrid(n_classes=n_classes, T=T, depth=8, dim=128, kernel=7, mult=2)


def mss_jamba_hybrid_large(n_classes: int = 15, T: int = 1000) -> MSSJambaHybrid:
    return MSSJambaHybrid(n_classes=n_classes, T=T, depth=6, dim=192, kernel=7, mult=2)


MAGJAM_MODELS: dict[str, Callable[..., MSSJambaHybrid]] = {
    "MAGJAM": mss_jamba_hybrid_d8,
    "MAGJAM_d4": mss_jamba_hybrid_d4,
}


__all__ = [
    "MagnitudePool1d",
    "ConcatPool1d",
    "MagGAP",
    "FastSSMBlock",
    "Mamba1DFast",
    "DropPath",
    "SE1D",
    "SwiGLUFFN",
    "MSSHybridBlock",
    "MSSJambaHybrid",
    "MAGJAMExtractor",
    "MAGJAMFullModel",
    "mss_jamba_hybrid",
    "mss_jamba_hybrid_d4",
    "mss_jamba_hybrid_d8",
    "mss_jamba_hybrid_large",
    "MAGJAM_MODELS",
]
