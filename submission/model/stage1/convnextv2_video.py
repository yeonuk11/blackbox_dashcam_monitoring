"""ConvNeXt V2-Nano video classifier for Stage-1 recapture detection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .convnextv2 import ConvNeXtV2, convnextv2_nano


class ConvNeXtV2VideoClassifier(nn.Module):
    """Classify a video from spatial frame features and temporal statistics.

    Shape flow is ``[B,T,3,H,W] -> [B*T,640] -> [B,T,640]``.  Temporal mean
    and population standard deviation are concatenated into ``[B,1280]``.
    Mean preserves persistent spatial recapture traces (moire/aliasing), while
    standard deviation exposes traces that fluctuate across captured frames.
    """

    def __init__(self, backbone: ConvNeXtV2, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = backbone
        pooled_dim = 2 * backbone.feature_dim
        self.temporal_norm = nn.LayerNorm(pooled_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled_dim, 1)

    def forward(self, clips: Tensor) -> Tensor:
        if clips.ndim != 5:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clips.shape)}")
        batch, frames, channels, height, width = clips.shape
        flat = clips.reshape(batch * frames, channels, height, width)
        frame_features = self.backbone.forward_features(flat)
        frame_features = frame_features.reshape(batch, frames, -1)
        temporal_mean = frame_features.mean(dim=1)
        temporal_std = frame_features.std(dim=1, unbiased=False)
        pooled = torch.cat((temporal_mean, temporal_std), dim=1)
        return self.classifier(self.dropout(self.temporal_norm(pooled))).squeeze(1)


def convnextv2_nano_video(*, dropout: float = 0.3, drop_path_rate: float = 0.1):
    backbone = convnextv2_nano(num_classes=1000, drop_path_rate=drop_path_rate)
    return ConvNeXtV2VideoClassifier(backbone, dropout=dropout)


__all__ = ["ConvNeXtV2VideoClassifier", "convnextv2_nano_video"]
