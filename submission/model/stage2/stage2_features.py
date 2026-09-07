"""Per-frame features for Stage 2, cached once per clip.

Stage 2 needs to answer *when* something happened to within +-0.3 s and *where* it
came from, over clips that run at 10, 15 and 30 fps.  The cache therefore fixes a
single **10 Hz sample grid in SECONDS** -- the same grid Stage 3 uses -- so the
model never sees a frame rate at all, and a prediction on the grid converts back to
the clip's own frame index with nothing but its fps.

Three feature blocks per sample, concatenated:

``cnn``   ImageNet ``convnext_tiny`` final-stage map, average-pooled to THREE
          HORIZONTAL BANDS x 768 channels.  Global average pooling would be the
          obvious choice and is wrong here: it destroys left/right information, and
          ``entry_side`` is 15% of the Stage-2 score.  The mirrored image is pushed
          through the same backbone and stored as ``cnn_flip`` so horizontal-flip
          augmentation is exact rather than a band swap that only approximates it.
``flow``  ``stage3_features.flow_features`` on a cv2 DIS field, reused unchanged so
          the two stages share one flow contract (and its ``flip_features``).  DIS
          at 384x288 is the configuration Stage 3 measured at r=+0.41 against CAN
          speed where ``raft_small`` returned r=+0.09 and 82% exact zeros.  The
          field is 2x average-pooled before the features are read off, which costs
          nothing measurable and makes the feature pass 7x cheaper (59 -> 419
          flows/s per thread).
``fd``    Frame-difference energy in five regions plus its 90th percentile.  This is
          the cheap half of the impact cue: contact is a whole-frame brightness
          discontinuity that survives even when the flow field gives up.

Everything except the backbone runs on CPU inside a worker process, so extraction
scales with cores and only the backbone touches the (shared) GPU.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s2common as common
import s2flow as S3F

HZ = 10.0                       # samples per second, everywhere
CNN_SIZE = (192, 192)           # w, h fed to the backbone
FLOW_SIZE = (384, 288)          # w, h DIS runs at (Stage 3's measured configuration)
FLOW_POOL = 2                   # the field is average-pooled by this before features
GRAY_SIZE = (192, 144)          # w, h the frame-difference block runs at
N_BANDS = 3                     # horizontal CNN bands: left / centre / right
BACKBONE = "convnext_tiny"
BACKBONE_CH = 768
WEIGHT_DIR = "models/pretrained"

FD_NAMES = ("fd_mean", "fd_p90", "fd_top", "fd_bottom", "fd_left", "fd_centre",
            "fd_right", "fd_max")

CNN_DIM = N_BANDS * BACKBONE_CH
FLOW_DIM = S3F.FEATURE_DIM
FD_DIM = len(FD_NAMES)
FEATURE_DIM = CNN_DIM + FLOW_DIM + FD_DIM

def configure(cnn_dim: int) -> int:
    """Point the module's dimension constants at a DIFFERENT CNN block.

    ``CNN_DIM`` is 3 bands x 768 for the frozen ImageNet extractor, but a cache
    written by a fine-tuned backbone (``stage2_backbone.FTBandExtractor``), or one
    that concatenates the fine-tuned and frozen blocks, is wider.  Everything
    downstream reads ``S2F.CNN_DIM`` / ``S2F.FEATURE_DIM`` at call time, so setting
    them once from the cache that is actually on disk keeps the trainer, the
    evaluator and the inference adapter in agreement without a second constant.
    ``stage2_long.Bank`` calls this from the first clip it loads.
    """
    global CNN_DIM, FEATURE_DIM
    CNN_DIM = int(cnn_dim)
    FEATURE_DIM = CNN_DIM + FLOW_DIM + FD_DIM
    return FEATURE_DIM


# ``fd`` regions that swap under a horizontal mirror.
_FD_SWAP = {FD_NAMES.index("fd_left"): FD_NAMES.index("fd_right"),
            FD_NAMES.index("fd_right"): FD_NAMES.index("fd_left")}


def fd_flip_perm() -> np.ndarray:
    perm = np.arange(FD_DIM, dtype=np.int64)
    for a, b in _FD_SWAP.items():
        perm[a] = b
    return perm


FD_FLIP_PERM = fd_flip_perm()


# ---------------------------------------------------------------------------
# Sample grid
# ---------------------------------------------------------------------------
@dataclass
class Window:
    """The seconds a clip contributes to the cache."""

    t0: float
    t1: float

    def times(self, hz: float = HZ) -> np.ndarray:
        n = max(1, int(round((self.t1 - self.t0) * hz)) + 1)
        return (self.t0 + np.arange(n, dtype=np.float64) / hz).astype(np.float64)


def clip_window(duration: float, event_sec: float | None,
                pre: float = 11.0, post: float = 5.0,
                max_full: float = 45.0) -> Window:
    """Whole clip when it is short enough, otherwise a window around the event.

    ``max_full`` is 45 s on purpose, which means every corpus on disk is cached
    WHOLE -- CCD 5 s, AI Hub 10 s, Nexar ~40 s.  Cutting Nexar to a window around
    its labelled ``time_of_event`` would have been 4x cheaper and would have
    silently leaked the answer into the evaluation: a model that can only point
    inside a 16 s window centred on the truth scores far better than the same model
    pointed at the whole clip, and the private Stage-2 folders are ~1250 frames
    (~42 s at the assumed 30 fps) -- Nexar-shaped, and uncut.  The window path is
    kept for any future corpus longer than ``max_full``.
    """
    if duration <= max_full or event_sec is None:
        return Window(0.0, max(0.0, duration - 1.0 / HZ))
    t0 = max(0.0, event_sec - pre)
    t1 = min(max(0.0, duration - 1.0 / HZ), event_sec + post)
    if t1 <= t0:
        t0, t1 = 0.0, max(0.0, duration - 1.0 / HZ)
    return Window(t0, t1)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _decode_at_times(path: str, times: Sequence[float], fps: float,
                     n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices_taken, BGR uint8 [T, H, W, 3] at native size)``.

    cv2 sequential read, not seeking: these clips are 5-40 s and a seek per sample
    costs more than decoding straight through.  The wanted frame indices are
    ``round(t * fps)`` clamped into the clip.
    """
    import cv2

    want = np.clip(np.round(np.asarray(times) * fps).astype(np.int64), 0,
                   max(0, n_frames - 1))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return want, np.zeros((0, 1, 1, 3), np.uint8)
    lo, hi = int(want.min()), int(want.max())
    if lo > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    keep: dict[int, np.ndarray] = {}
    need = set(want.tolist())
    i = lo
    while i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i in need:
            keep[i] = frame
        i += 1
    cap.release()
    if not keep:
        return want, np.zeros((0, 1, 1, 3), np.uint8)
    last = keep[min(keep)]
    out = []
    for w in want:
        f = keep.get(int(w))
        if f is None:
            f = last
        else:
            last = f
        out.append(f)
    return want, np.stack(out)


def _fd_features(gray: np.ndarray) -> np.ndarray:
    """``gray`` is uint8 [T, H, W]; returns [T, FD_DIM] float32 (sample 0 is zero)."""
    T, H, W = gray.shape
    out = np.zeros((T, FD_DIM), np.float32)
    if T < 2:
        return out
    d = np.abs(gray[1:].astype(np.int16) - gray[:-1].astype(np.int16)).astype(np.float32)
    h2, w3 = H // 2, W // 3
    cols = [d[:, :, :w3], d[:, :, w3:2 * w3], d[:, :, 2 * w3:]]
    vals = np.stack([
        d.mean(axis=(1, 2)),
        np.percentile(d.reshape(T - 1, -1), 90, axis=1),
        d[:, :h2].mean(axis=(1, 2)),
        d[:, h2:].mean(axis=(1, 2)),
        cols[0].mean(axis=(1, 2)),
        cols[1].mean(axis=(1, 2)),
        cols[2].mean(axis=(1, 2)),
        d.reshape(T - 1, -1).max(axis=1),
    ], axis=1)
    out[1:] = np.log1p(vals)
    return out


def clip_features_cpu(path: str, times: Sequence[float], fps: float, n_frames: int
                      ) -> dict | None:
    """Everything a worker can do without the GPU: decode, DIS flow, fd energy."""
    import cv2

    cv2.setNumThreads(1)          # this runs inside a pool worker; OpenCV must not fan out
    idx, bgr = _decode_at_times(path, times, fps, n_frames)
    if bgr.shape[0] == 0:
        return None
    T = bgr.shape[0]
    rgb = np.empty((T, CNN_SIZE[1], CNN_SIZE[0], 3), np.uint8)
    gflow = np.empty((T, FLOW_SIZE[1], FLOW_SIZE[0]), np.uint8)
    ggray = np.empty((T, GRAY_SIZE[1], GRAY_SIZE[0]), np.uint8)
    for i in range(T):
        f = bgr[i]
        rgb[i] = cv2.resize(f, CNN_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gflow[i] = cv2.resize(g, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        ggray[i] = cv2.resize(g, GRAY_SIZE, interpolation=cv2.INTER_AREA)
    del bgr

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fh, fw = FLOW_SIZE[1] // FLOW_POOL, FLOW_SIZE[0] // FLOW_POOL
    flows = np.zeros((T, 2, fh, fw), np.float32)
    prev = None
    for i in range(T):
        if prev is not None:
            fl = dis.calc(prev, gflow[i], None)          # [H, W, 2] px
            fl = fl.reshape(fh, FLOW_POOL, fw, FLOW_POOL, 2).mean(axis=(1, 3))
            flows[i] = fl.transpose(2, 0, 1)
        prev = gflow[i]
    import torch as _t

    _t.set_num_threads(1)
    # 2x-pooled field, so the quantiles and gradients cost 1/7 of what they do at
    # 384x288 (measured 419 vs 59 flows/s per thread) with no change in meaning:
    # every ROI is expressed as a fraction of the frame and the magnitudes stay in
    # the DIS output's own pixel units.
    flow_feat = S3F.flow_features(_t.from_numpy(flows)).numpy().astype(np.float32)
    return {"idx": idx.astype(np.int32), "rgb": rgb,
            "flow": flow_feat, "fd": _fd_features(ggray)}


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def weight_path(root: str | Path = ".") -> Path:
    return Path(root) / WEIGHT_DIR / f"{BACKBONE}_in1k.pt"


def save_backbone_weights(root: str | Path = ".") -> Path:
    """Download the ImageNet weights ONCE so the offline server never needs them."""
    import timm
    import torch

    p = weight_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p
    m = timm.create_model(BACKBONE, pretrained=True, features_only=True, out_indices=(3,))
    torch.save(m.state_dict(), p)
    return p


class BandExtractor:
    """``convnext_tiny`` -> ``[B, N_BANDS * 768]``, fp16, offline-first."""

    def __init__(self, device: str = "cuda", root: str | Path = ".",
                 batch: int = 32):
        import timm
        import torch

        self.torch = torch
        self.device = device
        self.batch = batch
        p = weight_path(root)
        pretrained = not p.exists()
        m = timm.create_model(BACKBONE, pretrained=pretrained, features_only=True,
                              out_indices=(3,))
        if p.exists():
            m.load_state_dict(torch.load(p, map_location="cpu"))
        else:
            save_backbone_weights(root)
        self.model = m.eval().to(device)
        if device.startswith("cuda"):
            self.model = self.model.half()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _forward(self, x):
        torch = self.torch
        with torch.no_grad():
            f = self.model(x)[-1]                     # [B, C, h, w]
            B, C, h, w = f.shape
            w3 = w // N_BANDS
            bands = [f[:, :, :, i * w3:(i + 1) * w3].mean(dim=(2, 3))
                     for i in range(N_BANDS)]
            # the right-most band absorbs the remainder columns, if any
            if w3 * N_BANDS != w:
                bands[-1] = f[:, :, :, (N_BANDS - 1) * w3:].mean(dim=(2, 3))
            return torch.stack(bands, dim=1).reshape(B, N_BANDS * C)

    def __call__(self, rgb: np.ndarray, flip: bool = False) -> np.ndarray:
        """``rgb`` uint8 [T, H, W, 3] -> float16 [T, N_BANDS*768]."""
        torch = self.torch
        out = np.empty((rgb.shape[0], CNN_DIM), np.float16)
        for i in range(0, rgb.shape[0], self.batch):
            chunk = rgb[i:i + self.batch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if flip:
                x = torch.flip(x, dims=[3])
            x = (x - self.mean) / self.std
            if self.device.startswith("cuda"):
                x = x.half()
            out[i:i + self.batch] = self._forward(x).float().cpu().numpy().astype(np.float16)
        return out


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str | Path, corpus: str, cid: str) -> Path:
    return Path(cache_dir) / corpus / f"{cid}.npz"


def save_clip(cache_dir: str | Path, corpus: str, cid: str, times: np.ndarray,
              idx: np.ndarray, cnn: np.ndarray, cnn_flip: np.ndarray,
              flow: np.ndarray, fd: np.ndarray, meta: dict) -> Path:
    p = cache_path(cache_dir, corpus, cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, t=times.astype(np.float32), idx=idx.astype(np.int32),
             cnn=cnn.astype(np.float16), cnn_flip=cnn_flip.astype(np.float16),
             flow=flow.astype(np.float16), fd=fd.astype(np.float16),
             meta=np.frombuffer(repr(meta).encode("utf-8"), dtype=np.uint8))
    return p


def load_clip(cache_dir: str | Path, corpus: str, cid: str) -> dict | None:
    p = cache_path(cache_dir, corpus, cid)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in ("t", "idx", "cnn", "cnn_flip", "flow", "fd")}
    except Exception:
        return None


def assemble(d: dict, flip: bool = False) -> np.ndarray:
    """One cached clip -> ``[T, FEATURE_DIM]`` float32, optionally mirrored."""
    cnn = d["cnn_flip"] if flip else d["cnn"]
    flow = d["flow"].astype(np.float32)
    fd = d["fd"].astype(np.float32)
    if flip:
        flow = S3F.flip_features(flow)
        fd = fd[:, FD_FLIP_PERM]
    return np.concatenate([cnn.astype(np.float32), flow, fd], axis=1)


__all__ = ["HZ", "FEATURE_DIM", "configure", "CNN_DIM", "FLOW_DIM", "FD_DIM", "Window",
           "clip_window", "clip_features_cpu", "BandExtractor", "save_backbone_weights",
           "weight_path", "cache_path", "save_clip", "load_clip", "assemble"]
