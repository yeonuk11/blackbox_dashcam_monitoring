"""ConvNeXt V2 image backbone used for Stage-1 domain adaptation.

The module names and tensor operations intentionally match Meta's official
ConvNeXt V2 implementation so the official ImageNet-1K checkpoint can be
loaded strictly without key conversion.

Reference:
    Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders", 2023, Section 3 (FCMAE and Global Response Normalization).
    Official code: https://github.com/facebookresearch/ConvNeXt-V2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth with no checkpoint parameters."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    """LayerNorm supporting NHWC and NCHW layouts.

    Input/output dimensions are either ``[B, H, W, C]`` (channels_last) or
    ``[B, C, H, W]`` (channels_first).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2, Section 3.2.

    The official equation is ``X + gamma * (X * Nx) + beta``, where ``Gx`` is
    the spatial L2 norm and ``Nx`` is Gx normalized across channels.
    Input/output: ``[B, H, W, C]``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # [B, 1, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXt V2 residual block.

    Shape flow: ``[B,C,H,W] -> [B,H,W,C] -> [B,C,H,W]``. The NHWC section
    allows the two point-wise convolutions to be represented by Linear layers,
    exactly as in the official checkpoint.
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # Linear/GRN operate on channel-last [B,H,W,C].
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # Restore PyTorch convolution layout.
        return residual + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2 classifier with an exposed 2-D feature path."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt V2 requires exactly four stages")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for stage_index, depth in enumerate(depths):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[stage_index], drop_path=rates[offset + block_index])
                    for block_index in range(depth)
                ]
            )
            self.stages.append(stage)
            offset += depth

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.feature_dim = int(dims[-1])
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: Tensor) -> Tensor:
        """Map ``[B,3,H,W]`` to pooled embeddings ``[B,C]``."""
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
        x = x.mean(dim=(-2, -1))  # Global average pool [B,C,H,W] -> [B,C].
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))


def convnextv2_nano(**kwargs: Any) -> ConvNeXtV2:
    """Build the official 15.6M-parameter ConvNeXt V2-Nano topology."""
    return ConvNeXtV2(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), **kwargs)


def load_meta_checkpoint(
    model: ConvNeXtV2,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load an official Meta checkpoint while keeping training fully offline."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state: Mapping[str, Tensor]
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        raise TypeError(f"unsupported checkpoint object: {type(checkpoint)!r}")
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


__all__ = ["ConvNeXtV2", "convnextv2_nano", "load_meta_checkpoint"]
