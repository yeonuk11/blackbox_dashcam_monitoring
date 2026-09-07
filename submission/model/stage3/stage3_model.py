"""Stage 3 temporal model.

The flow features of stage3_features are per-0.1 s snapshots; the labels are not.
"ACCELERATING" is a statement about a ~0.5 s window of speed and "LEFT" about a
steering angle that persists for seconds, so the model's whole job is temporal
integration. A dilated 1-D CNN does that with a receptive field of several seconds at a
fraction of the cost of an RNN, and -- unlike an RNN -- it is fully parallel over T,
which matters because the eval server has to label ~150 clips inside a 60 minute budget
shared with two other stages.

Three things are bundled with the weights so inference needs no extra files:
  * feature mean/std buffers   (normalisation is part of the model)
  * a label transition matrix  (Viterbi smoothing; raw per-frame argmax flickers)
  * per-class logit offsets    (macro-F1's optimum is not argmax under class imbalance)
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACCEL = 4
N_STEER = 3
AUX_HEADS = ("speed", "accel_mps2", "steer_deg")


class _ResBlock(nn.Module):
    """Dilated residual block: two convs, GroupNorm, GELU, identity shortcut."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3, groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(groups, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(groups, ch), ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(self.norm2(self.conv2(h)))
        return F.gelu(x + h)


class Stage3Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 4,
                 dilations: Sequence[int] = (1, 2, 4, 8), dropout: float = 0.1,
                 feature_names: Sequence[str] | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.feature_names = list(feature_names) if feature_names else []
        # Normalisation lives in the module so a checkpoint is self-contained.
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
        )
        dil = list(dilations) if dilations else [1]
        self.dilations = list(dil)
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden, dil[i % len(dil)], dropout=dropout) for i in range(self.layers)]
        )
        self.head_accel = nn.Conv1d(hidden, N_ACCEL, 1)
        self.head_steer = nn.Conv1d(hidden, N_STEER, 1)
        self.head_aux = nn.Conv1d(hidden, len(AUX_HEADS), 1)

    # -- normalisation -------------------------------------------------------------
    @torch.no_grad()
    def fit_normalization(self, feats: np.ndarray | torch.Tensor) -> None:
        """``feats`` is [N, D] (all training samples stacked)."""
        x = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(np.asarray(feats))
        x = x.reshape(-1, self.in_dim).float()
        m = x.mean(0)
        s = x.std(0).clamp_min(1e-3)
        self.feat_mean.copy_(torch.nan_to_num(m))
        self.feat_std.copy_(torch.nan_to_num(s, nan=1.0).clamp_min(1e-3))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``: [B, T, D] raw features -> dict of [B, T, C]."""
        x = torch.nan_to_num(x.float())
        x = (x - self.feat_mean) / self.feat_std
        x = x.clamp(-8.0, 8.0).transpose(1, 2)          # [B, D, T]
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        aux = self.head_aux(h).transpose(1, 2)
        return {
            "accel": self.head_accel(h).transpose(1, 2),
            "steer": self.head_steer(h).transpose(1, 2),
            "speed": aux[..., 0],
            "accel_mps2": aux[..., 1],
            "steer_deg": aux[..., 2],
        }


# --------------------------------------------------------------------------------------
# Sequence smoothing
# --------------------------------------------------------------------------------------
def estimate_transition(sequences: Sequence[Sequence[int]], n_classes: int,
                        smoothing: float = 1.0) -> np.ndarray:
    """Log transition matrix from training label sequences (rows sum to 1 in prob space)."""
    cnt = np.full((n_classes, n_classes), float(smoothing), dtype=np.float64)
    for seq in sequences:
        a = np.asarray(seq, dtype=np.int64)
        if a.size < 2:
            continue
        np.add.at(cnt, (a[:-1], a[1:]), 1.0)
    p = cnt / cnt.sum(axis=1, keepdims=True)
    return np.log(np.maximum(p, 1e-12)).astype(np.float64)


def viterbi_decode(logprobs: np.ndarray, transition: np.ndarray, min_dwell: int = 1,
                   trans_scale: float = 1.0) -> np.ndarray:
    """Best label path over the T axis.

    ``logprobs`` [T, C] (any monotone score works, log-softmax is the natural choice),
    ``transition`` [C, C] in log space, ``min_dwell`` the minimum number of consecutive
    samples a label must hold. The dwell constraint is enforced by expanding the state
    space to (class, age) pairs -- ages 0..min_dwell-2 may only advance within the same
    class, and only the saturated age may switch -- which is exact, unlike the usual
    median-filter approximation.
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    T, C = lp.shape
    tr = np.asarray(transition, dtype=np.float64) * float(trans_scale)
    if tr.shape != (C, C):
        tr = np.zeros((C, C), dtype=np.float64)
    D = max(1, int(min_dwell))
    S = C * D
    NEG = -1e18

    big = np.full((S, S), NEG, dtype=np.float64)
    for c in range(C):
        for d in range(D):
            s = c * D + d
            if d < D - 1:
                big[s, c * D + d + 1] = 0.0        # must stay in class c
            else:
                big[s, c * D + D - 1] = tr[c, c]   # stay, saturated
                for c2 in range(C):
                    if c2 != c:
                        big[s, c2 * D + 0] = tr[c, c2]
    if D == 1:
        big = tr.copy()

    emis = np.repeat(lp, D, axis=1) if D > 1 else lp        # [T, S], state (c,d) -> lp[:, c]
    score = emis[0].copy()
    if D > 1:
        # a path may only start at age 0 of some class
        mask = np.zeros(S, dtype=bool)
        mask[np.arange(C) * D] = True
        score = np.where(mask, score, NEG)
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        m = score[:, None] + big
        back[t] = np.argmax(m, axis=0)
        score = m[back[t], np.arange(S)] + emis[t]
    path = np.zeros(T, dtype=np.int64)
    s = int(np.argmax(score))
    for t in range(T - 1, -1, -1):
        path[t] = s // D if D > 1 else s
        s = int(back[t][s])
    return path


# --------------------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------------------
def tune_class_bias(logits: np.ndarray, labels: np.ndarray,
                    metric_fn: Callable[[np.ndarray, np.ndarray], float],
                    grid: Sequence[float] | None = None, rounds: int = 3,
                    init: np.ndarray | None = None,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Coordinate ascent over per-class logit offsets that maximises ``metric_fn``.

    Macro-F1 weights every class equally, so under the heavy class imbalance of Stage 3
    (CONSTANT dominates, ACCELERATING/DECELERATING are rare) the argmax decision rule is
    not the optimum -- shifting a rare class up by a fraction of a logit trades a little
    precision for a lot of recall on that class and raises the macro average.

    ``mask`` optionally restricts the metric to a subset of rows (Stage 3 scores steer
    only where the ground-truth accel is not STOPPED) while all rows still get a label.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else 1,), dtype=np.float32)
    C = x.shape[1]
    b = np.zeros(C, dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64).copy()
    g = list(grid) if grid is not None else [-1.5, -1.0, -0.7, -0.45, -0.25, -0.1, 0.0,
                                             0.1, 0.25, 0.45, 0.7, 1.0, 1.5, 2.0]
    sel = slice(None) if mask is None else np.asarray(mask, dtype=bool)

    def score(bv: np.ndarray) -> float:
        pred = np.argmax(x + bv, axis=1)
        try:
            return float(metric_fn(y[sel], pred[sel]))
        except Exception:
            return -1.0

    best = score(b)
    for _ in range(max(1, int(rounds))):
        improved = False
        for c in range(C):
            base = b[c]
            for v in g:
                b[c] = base + v
                s = score(b)
                if s > best + 1e-9:
                    best, base, improved = s, b[c], True
            b[c] = base
        if not improved:
            break
    return b.astype(np.float32)


# --------------------------------------------------------------------------------------
# Checkpoint bundle: everything inference.py needs, in one file.
# --------------------------------------------------------------------------------------
def save_bundle(path, model: Stage3Net, feature_names: Sequence[str],
                trans_accel: np.ndarray, trans_steer: np.ndarray,
                bias_accel: np.ndarray, bias_steer: np.ndarray,
                min_dwell: dict[str, int] | None = None, extra: dict | None = None,
                members: Sequence[dict] | None = None) -> None:
    """``members`` are *additional* state dicts of the same architecture.

    A Stage 3 net is 12 MB and its forward pass is 3 ms per 600-sample clip, while the
    optical flow in front of it costs seconds -- so seed-averaging is nearly free at
    inference time, but only if the extra weights actually reach the evaluation server.
    ``scripts/build_submission.py`` ships two Stage 3 files by name (best.pt, aux.pt), so
    the members ride *inside* the bundle instead of beside it: one file, same name, same
    packaging rules, and an older loader simply ignores the key.
    """
    import os

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    mem = [{k: v.detach().cpu() for k, v in sd.items()} for sd in (members or [])]
    torch.save({
        "state_dict": model.state_dict(),
        "members": mem,
        "arch": {"in_dim": model.in_dim, "hidden": model.hidden, "layers": model.layers,
                 "dilations": list(model.dilations)},
        "feature_names": list(feature_names),
        "feat_mean": model.feat_mean.detach().cpu().numpy(),
        "feat_std": model.feat_std.detach().cpu().numpy(),
        "trans_accel": np.asarray(trans_accel, dtype=np.float32),
        "trans_steer": np.asarray(trans_steer, dtype=np.float32),
        "bias_accel": np.asarray(bias_accel, dtype=np.float32),
        "bias_steer": np.asarray(bias_steer, dtype=np.float32),
        "min_dwell": dict(min_dwell or {"accel": 2, "steer": 2}),
        "extra": dict(extra or {}),
    }, str(path))


def load_bundle(path, device: str = "cpu") -> tuple[Stage3Net, dict]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    a = ck["arch"]
    net = Stage3Net(a["in_dim"], hidden=a["hidden"], layers=a["layers"],
                    dilations=a.get("dilations") or (1, 2, 4, 8),
                    feature_names=ck.get("feature_names"))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, ck


def build_members(ck: dict, primary: Stage3Net, device: str = "cpu") -> list[Stage3Net]:
    """``[primary] + one net per extra state dict in the bundle``.

    Every member shares the bundle's ``arch``; a member that does not load is dropped
    rather than raised, because on the evaluation server a broken seed must cost its own
    share of the average and nothing else.
    """
    nets = [primary]
    a = ck.get("arch") or {}
    for i, sd in enumerate(ck.get("members") or []):
        try:
            n = Stage3Net(a["in_dim"], hidden=a["hidden"], layers=a["layers"],
                          dilations=a.get("dilations") or (1, 2, 4, 8),
                          feature_names=ck.get("feature_names"))
            n.load_state_dict(sd)
            nets.append(n.to(device).eval())
        except Exception:
            continue
    return nets


__all__ = [
    "Stage3Net", "N_ACCEL", "N_STEER", "AUX_HEADS",
    "estimate_transition", "viterbi_decode", "tune_class_bias",
    "save_bundle", "load_bundle", "build_members",
]
