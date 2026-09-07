"""A Stage-2 visual backbone that is TRAINED on impact, instead of borrowed from ImageNet.

WHY
---
``stage2_features.BandExtractor`` runs a frozen ``convnext_tiny`` trained to name
1000 object categories, and everything downstream sees only what that
representation kept.  The out-of-fold diagnostic says that is where Stage 2 loses:
on the 40 s stitched Nexar protocol the model puts the collision in the RIGHT 5 s
chunk only 60% of the time, and *given* the right chunk it is inside +-0.3 s 77% of
the time.  The bottleneck is therefore DETECTION -- "is an impact happening in this
frame" -- not fine localisation, and detection is exactly the question an ImageNet
classifier was never asked.

WHAT
----
``ImpactNet`` is the same ``convnext_tiny`` with two changes:

* **A 6-channel stem.**  One sample is the pair (frame_t, frame_{t-1}) on the same
  10 Hz grid the cache already uses, stacked on the channel axis, so the first conv
  can form a temporal difference.  This costs NOTHING at inference: sample t-1 is
  the previous grid sample, which ``predict_folder`` has already decoded for its own
  ``fd`` and ``flow`` blocks.  The stem is inflated as ``[W, 0]`` so at
  initialisation the network computes exactly the ImageNet response to frame t and
  the motion path grows from zero.
* **A supervised head that predicts WHERE IN THE ACCIDENT this frame sits**: a
  13-bin classification of the signed offset to the collision (and, where labelled,
  to the entry), plus the two clip-level categoricals near the event.  Binary
  "impact/not" would collapse the approach phase, which is what ``entry`` needs.

The pooled output keeps the three horizontal bands of the frozen extractor -- and
for the same reason: global average pooling destroys left/right, and ``entry_side``
is 15% of Stage 2.  ``adaptive_avg_pool2d`` to exactly ``(1, N_BANDS)`` replaces the
frozen path's integer column split, which at an 8-wide map gave bands of 2, 2 and 4
columns.

Inference cost is not a constraint here.  Measured on a real 1250-frame folder, the
frozen backbone is 0.04 s of ``predict_folder``'s 6.05 s (DIS optical flow is 3.05 s,
JPEG read + resize 1.48 s, flow features 1.16 s).  Running a bigger backbone at
256x256 instead of 192x192 costs about 0.1 s more per folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_BANDS = 3
BACKBONE_CH = {"convnext_tiny": 768, "convnext_nano": 640, "convnext_small": 768}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Signed offset (seconds) of this frame from the event, as 13 ordered bins.  The
# inner bins are +-0.3 s wide because that is the width Stage 2 is scored at; the
# outer ones are wide because "far before" and "far after" only need to be told
# apart from each other.
BIN_EDGES = (-4.0, -2.5, -1.6, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.6, 2.5, 4.0)
N_BINS = len(BIN_EDGES) + 1
HIT_BIN = 6                      # the bin containing |offset| <= 0.3 s


def offset_bin(dt: np.ndarray | float) -> np.ndarray:
    return np.searchsorted(np.asarray(BIN_EDGES), np.asarray(dt), side="right")


class ImpactNet(nn.Module):
    def __init__(self, name: str = "convnext_tiny", n_frames: int = 2,
                 weights: str | Path | None = None, drop_path: float = 0.1,
                 head_hidden: int = 512, pool: str = "mean"):
        super().__init__()
        import timm

        self.name = name
        self.n_frames = int(n_frames)
        self.pool = pool
        ch = BACKBONE_CH.get(name, 768)
        self.ch = ch
        m = timm.create_model(name, pretrained=False, features_only=True,
                              out_indices=(3,), drop_path_rate=drop_path)
        if weights is not None and Path(weights).exists():
            sd = torch.load(weights, map_location="cpu")
            missing = m.load_state_dict(sd, strict=False)
            if getattr(missing, "unexpected_keys", None):
                print(f"[stage2_backbone] unexpected keys: {missing.unexpected_keys[:4]}")
        self.backbone = m
        self._inflate_stem()
        self.feat_dim = ch * N_BANDS * (2 if pool == "meanmax" else 1)
        self.norm = nn.LayerNorm(self.feat_dim)
        self.trunk = nn.Sequential(nn.Linear(self.feat_dim, head_hidden), nn.GELU(),
                                   nn.Dropout(0.1))
        self.head_col = nn.Linear(head_hidden, N_BINS)
        self.head_ent = nn.Linear(head_hidden, N_BINS)
        self.head_side = nn.Linear(head_hidden, 2)
        self.head_evas = nn.Linear(head_hidden, 2)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).repeat(self.n_frames)
                             .view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).repeat(self.n_frames)
                             .view(1, -1, 1, 1))

    # -- stem ----------------------------------------------------------------
    def _inflate_stem(self) -> None:
        """3 -> 3*n_frames input channels, as ``[W, 0, 0, ...]``.

        At initialisation the network's response to (frame_t, frame_{t-1}) is bit for
        bit its ImageNet response to frame_t alone, so fine-tuning starts from the
        frozen baseline rather than from a blurred average of two frames.
        """
        if self.n_frames == 1:
            return
        conv = None
        for mod in self.backbone.modules():
            if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                conv = mod
                break
        if conv is None:
            raise RuntimeError("no 3-channel stem conv found")
        w = conv.weight.data
        new = torch.zeros(w.shape[0], 3 * self.n_frames, *w.shape[2:], dtype=w.dtype)
        new[:, :3] = w
        conv.weight = nn.Parameter(new)
        conv.in_channels = 3 * self.n_frames

    # -- features ------------------------------------------------------------
    def bands(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [B, 3*n_frames, H, W] already normalised -> [B, feat_dim]."""
        f = self.backbone(x)[-1]                              # [B, C, h, w]
        a = F.adaptive_avg_pool2d(f, (1, N_BANDS)).flatten(1)
        if self.pool == "meanmax":
            b = F.adaptive_max_pool2d(f, (1, N_BANDS)).flatten(1)
            return torch.cat([a, b], dim=1)
        return a

    def normalize(self, u8: torch.Tensor) -> torch.Tensor:
        """uint8 [B, H, W, 3*n_frames] or float [B, C, H, W] -> normalised float."""
        if u8.dtype == torch.uint8:
            u8 = u8.permute(0, 3, 1, 2).float().div_(255.0)
        return (u8 - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> dict:
        z = self.trunk(self.norm(self.bands(x)))
        return {"col": self.head_col(z), "ent": self.head_ent(z),
                "side": self.head_side(z), "evasion": self.head_evas(z)}


# --------------------------------------------------------------------------------------
# Inference-time extractor -- the mirror of stage2_features.BandExtractor
# --------------------------------------------------------------------------------------
class FTBandExtractor:
    """Fine-tuned backbone -> ``[T, feat_dim]`` float16 for a whole clip.

    ``rgb`` is the clip's frames on the 10 Hz grid.  Sample ``i`` is the pair
    ``(rgb[i], rgb[i-1])`` with ``rgb[-1]`` replaced by ``rgb[0]``: sample 0 is
    therefore degenerate in exactly the way the cached ``fd`` and ``flow`` blocks
    already are, which is what ``stage2_long.MIN_START_IDX`` exists to exclude.
    """

    def __init__(self, ckpt: str | Path, device: str = "cuda", batch: int = 64,
                 size: int = 256):
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        self.net = ImpactNet(name=cfg["name"], n_frames=cfg["n_frames"],
                             weights=None, drop_path=0.0,
                             head_hidden=cfg.get("head_hidden", 512),
                             pool=cfg.get("pool", "mean"))
        self.net.load_state_dict(ck["model"])
        self.net = self.net.eval().to(device)
        self.half = device.startswith("cuda")
        if self.half:
            self.net = self.net.half()
        self.device = device
        self.batch = batch
        self.size = int(cfg.get("size", size))
        self.n_frames = cfg["n_frames"]
        self.feat_dim = self.net.feat_dim

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, flip: bool = False) -> np.ndarray:
        """``rgb`` uint8 [T, S, S, 3] on the grid -> float16 [T, feat_dim]."""
        T = rgb.shape[0]
        out = np.empty((T, self.feat_dim), np.float16)
        prev_ix = np.maximum(np.arange(T) - 1, 0)
        for s in range(0, T, self.batch):
            e = min(T, s + self.batch)
            cur = torch.from_numpy(np.ascontiguousarray(rgb[s:e])).to(self.device)
            pre = torch.from_numpy(np.ascontiguousarray(rgb[prev_ix[s:e]])).to(self.device)
            x = torch.cat([cur, pre], dim=3) if self.n_frames == 2 else cur
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if flip:
                x = torch.flip(x, dims=[3])
            x = (x - self.net.mean) / self.net.std
            if self.half:
                x = x.half()
            out[s:e] = self.net.bands(x).float().cpu().numpy().astype(np.float16)
        return out


__all__ = ["ImpactNet", "FTBandExtractor", "BIN_EDGES", "N_BINS", "HIT_BIN",
           "offset_bin", "N_BANDS"]
