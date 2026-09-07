"""Stage 2 temporal model: two per-frame heatmaps and two clip-level classifiers.

Stage 2 is scored ``0.35*Acc@+-0.3s(collision) + 0.35*Acc@+-0.3s(entry)
+ 0.15*F1(entry_side) + 0.15*F1(evasion_space)``.  Two design consequences run
through this file:

* **Timing is hit/miss, not MAE.**  The heads are therefore trained as a
  distribution over time -- soft cross-entropy against a Gaussian centred on the
  label -- and decoded by argmax.  A regression head would minimise the average
  error and cheerfully sit 0.4 s off on every clip; what this scoring wants is mass
  inside a 0.6 s window.  The Gaussian's sigma is given in SECONDS and converted
  with the clip's own sample rate, because the corpora run at 10 / 15 / 30 fps.
* **Entry precedes contact.**  ``decode`` takes the collision argmax first (it is
  the better-supported of the two -- every corpus labels it) and then restricts the
  entry argmax to ``<= collision``.  The reverse order would let a confident entry
  drag the collision earlier and lose both.

The trunk is a bidirectional dilated 1-D CNN.  It is bidirectional by construction
(symmetric padding, no causal masking): the collision frame is far easier to place
from the aftermath than from the approach, and nothing here runs online.  Six
blocks at dilations 1,2,4,8,16,32 over a 10 Hz grid give a receptive field of about
25 s, i.e. the whole clip.

The clip-level heads do not pool the whole sequence uniformly.  ``entry_side`` is a
statement about the moment of entry and ``evasion_space`` about the moment of
impact, so the pooled vector concatenates a mean, a max, a learned-query attention
pool, and two event-anchored pools that use the model's OWN heatmaps as weights.
Normalisation buffers live in the module so a checkpoint is self-contained.
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE_LABELS = ("LEFT", "RIGHT")
EVASION_LABELS = ("0", "1")
HEATMAP_SIGMA_SEC = 0.15        # +-0.3 s tolerance -> a 2-sigma half-width
NEG_INF = -1e4


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


class Stage2Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
                 dropout: float = 0.1, proj: int = 512,
                 feature_meta: dict | None = None):
        super().__init__()
        # Which CNN block this net was trained on, copied from the feature cache that
        # produced it.  The fine-tuned 3-band block and the frozen ImageNet one are both
        # 2304 wide, so nothing else in the checkpoint distinguishes them, and feeding a
        # frozen-feature net fine-tuned features (or the reverse) is a silent wrong answer
        # on every folder.  Absent -> frozen, which is what every checkpoint written
        # before the fine-tuned pipeline existed was trained on.
        self.feature_meta = dict(feature_meta or {})
        self.wants_ft = bool(self.feature_meta.get("backbone"))
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.dilations = list(dilations)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

        # The 2304-d CNN block dominates the input, so it gets its own bottleneck
        # before the trunk instead of a single wide 1x1 over all 2382 channels.
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, proj, 1),
            nn.GroupNorm(min(8, proj), proj),
            nn.GELU(),
            nn.Conv1d(proj, hidden, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden, d, dropout=dropout)
                                     for d in self.dilations])
        self.head_collision = nn.Conv1d(hidden, 1, 1)
        self.head_entry = nn.Conv1d(hidden, 1, 1)
        self.att_query = nn.Conv1d(hidden, 1, 1)
        pooled = hidden * 5
        self.clip_mlp = nn.Sequential(
            nn.Linear(pooled, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head_side = nn.Linear(hidden, len(SIDE_LABELS))
        self.head_evasion = nn.Linear(hidden, len(EVASION_LABELS))

    # -- normalisation -----------------------------------------------------------
    @torch.no_grad()
    def fit_normalization(self, feats: np.ndarray | torch.Tensor) -> None:
        x = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(np.asarray(feats))
        x = x.reshape(-1, self.in_dim).float()
        m = torch.nan_to_num(x.mean(0))
        s = torch.nan_to_num(x.std(0), nan=1.0).clamp_min(1e-3)
        self.feat_mean.copy_(m)
        self.feat_std.copy_(s)

    # -- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """``x`` [B, T, D] raw features, ``mask`` [B, T] bool (True = real sample)."""
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        x = torch.nan_to_num(x.float())
        x = (x - self.feat_mean) / self.feat_std
        x = x.clamp(-8.0, 8.0).transpose(1, 2)                 # [B, D, T]
        x = x * mask.unsqueeze(1).float()
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        h = h * mask.unsqueeze(1).float()

        col = self.head_collision(h).squeeze(1)                # [B, T]
        ent = self.head_entry(h).squeeze(1)
        col = col.masked_fill(~mask, NEG_INF)
        ent = ent.masked_fill(~mask, NEG_INF)

        att = self.att_query(h).squeeze(1).masked_fill(~mask, NEG_INF)
        n = mask.float().sum(1, keepdim=True).clamp_min(1.0)
        mean = h.sum(2) / n
        mx = h.masked_fill(~mask.unsqueeze(1), NEG_INF).max(dim=2).values
        pools = [mean, mx]
        for logits in (att, col, ent):
            w = torch.softmax(logits, dim=1).unsqueeze(1)       # [B, 1, T]
            pools.append((h * w).sum(2))
        z = self.clip_mlp(torch.cat(pools, dim=1))
        return {"collision": col, "entry": ent,
                "side": self.head_side(z), "evasion": self.head_evasion(z)}


# --------------------------------------------------------------------------------------
# Targets and losses
# --------------------------------------------------------------------------------------
def gaussian_target(t_sec: np.ndarray, centre_sec: float,
                    sigma_sec: float = HEATMAP_SIGMA_SEC) -> np.ndarray:
    """Normalised Gaussian over the clip's own sample times, in SECONDS.

    Every corpus is cached on the same 10 Hz grid, but the grid is built from each
    clip's real fps, so passing seconds (not sample counts) is what keeps sigma
    physically the same width on a 10 fps CCD clip and a 30 fps Nexar clip.
    """
    d = (np.asarray(t_sec, dtype=np.float64) - float(centre_sec)) / max(1e-6, sigma_sec)
    g = np.exp(-0.5 * d * d)
    s = g.sum()
    if s <= 1e-9:                       # centre outside the window -> nearest sample
        g = np.zeros_like(g)
        g[int(np.argmin(np.abs(np.asarray(t_sec) - centre_sec)))] = 1.0
        return g.astype(np.float32)
    return (g / s).astype(np.float32)


def soft_time_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of a distribution over time against a soft target.

    ``logits`` [B, T], ``target`` [B, T] (sums to 1 over valid samples), ``mask``
    [B, T], ``weight`` [B] -- the per-clip label confidence.  Returns a scalar.
    """
    logp = torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=1)
    per = -(target * logp).sum(dim=1)
    w = weight.clamp_min(0.0)
    return (per * w).sum() / w.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------
def _smooth(p: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing along the last axis, in SAMPLES."""
    if sigma <= 0:
        return p
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(p, ((0, 0), (r, r)), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, pad)


def _entry_max_lead_default() -> float:
    """``stage2_entry.ENTRY_MAX_LEAD_SEC``, or 0 (no bound) if that module is absent.

    One constant, one place.  The import is guarded because ``stage2_model`` must keep
    working on the evaluation server even if the entry module failed to be staged: a
    missing bound is the behaviour that shipped in submissions #1-#4, not a crash.
    """
    try:
        import stage2_entry

        return float(getattr(stage2_entry, "ENTRY_MAX_LEAD_SEC", 0.0))
    except Exception:                                           # noqa: BLE001
        return 0.0


def decode(collision_logits: np.ndarray, entry_logits: np.ndarray,
           lengths: Sequence[int], t_sec: Sequence[np.ndarray],
           smooth_sigma: float = 0.0, tol: float = 0.3,
           mode: str = "window",
           entry_max_lead_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``[B, T]`` logits -> ``(collision_sec, entry_sec)`` with entry <= collision.

    ``entry_max_lead_sec`` bounds how far before the collision the entry argmax may look.
    It DEFAULTS TO 0 -- unbounded, exactly what submissions #1-#4 did -- so that every
    evaluation script that calls this function keeps producing the numbers it produced
    before; the submission path opts in explicitly in ``predict_folder``.  The bound is a
    restriction of the search, not a clamp of the answer, so a clip whose entry mass peaks
    inside the window is untouched.  Measured with the shipped 3-net ensemble on
    aihub-long (the only view with any entry target at all, and it is derived): entry Acc
    0.7047 unbounded -> 0.6982 at 2.5 s, moving 3.0% of aihub-long / 9.5% of nexar-long /
    5.1% of ccd-long answers.  See ``stage2_entry`` for why 2.5 s and for the leaderboard
    probe that is the only thing that can actually settle the question.

    ``mode="window"`` is the decision rule this metric actually asks for.  Stage 2
    pays 1 for ``|pred - true| <= 0.3 s`` and 0 otherwise, so the answer that
    maximises the expected score is the time that maximises the probability MASS
    inside its own +-0.3 s window, i.e. ``argmax_i sum_j p_j [|t_j - t_i| <= tol]``
    -- a boxcar correlation, not a peak.  It differs from the plain argmax whenever
    the heatmap is broad or bimodal, which is exactly when the peak is least
    trustworthy.  ``mode="argmax"`` keeps the naive rule for comparison.

    Neither rule takes an expectation: the mean of a bimodal heatmap lands between
    the two modes, where the answer certainly is not, and a near miss scores zero.
    """
    B = collision_logits.shape[0]
    pc = _softmax_np(collision_logits, lengths)
    pe = _softmax_np(entry_logits, lengths)
    if smooth_sigma > 0:
        pc = _smooth(pc, smooth_sigma)
        pe = _smooth(pe, smooth_sigma)
    c_out = np.zeros(B, np.float64)
    e_out = np.zeros(B, np.float64)
    max_lead = float(entry_max_lead_sec)
    for i in range(B):
        n = int(lengths[i])
        ts = np.asarray(t_sec[i], dtype=np.float64)[:n]
        vc, ve = pc[i, :n], pe[i, :n]
        if mode == "window" and n > 1:
            dt = float(np.median(np.diff(ts))) if n > 1 else 0.1
            k = int(math.floor(tol / max(dt, 1e-6) + 1e-6))
            if k > 0:
                box = np.ones(2 * k + 1)
                vc = np.convolve(np.pad(vc, k, mode="constant"), box, mode="valid")
                ve = np.convolve(np.pad(ve, k, mode="constant"), box, mode="valid")
        ci = int(np.argmax(vc))
        # The bound is applied in TIME, not in samples: the sample grid is 384 points over
        # whatever duration `common.s2_assumed_fps` implies, so a fixed sample count would
        # mean a different number of seconds on a folder of a different length.
        lo = 0
        if max_lead > 0 and n > 1:
            lo = int(np.searchsorted(ts, ts[ci] - max_lead - 1e-9, side="left"))
            lo = max(0, min(lo, ci))
        ei = lo + int(np.argmax(ve[lo:ci + 1]))
        c_out[i] = ts[min(ci, len(ts) - 1)]
        e_out[i] = ts[min(ei, len(ts) - 1)]
    return c_out, e_out


def _softmax_np(logits: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    out = np.zeros_like(logits, dtype=np.float64)
    for i in range(logits.shape[0]):
        n = int(lengths[i])
        v = logits[i, :n].astype(np.float64)
        v = v - v.max()
        e = np.exp(v)
        out[i, :n] = e / e.sum()
    return out


__all__ = ["Stage2Net", "SIDE_LABELS", "EVASION_LABELS", "HEATMAP_SIGMA_SEC",
           "gaussian_target", "soft_time_ce", "decode"]


# --------------------------------------------------------------------------------------
# Inference glue
#
# stage2_infer probes this module for one of ("predict_folder", "predict_one",
# "predict_item") and falls back to its OpenCV heuristic when none exists. The first
# submission shipped without this function, so the trained network never ran on the
# leaderboard at all -- Stage 2's 0.238 was the heuristic's score, not the model's.
#
# The pipeline mirrors training exactly: sample ~HZ frames per second, build the
# [cnn | flow | fd] blocks that stage2_features.assemble() concatenates, run the net over
# the whole sequence at once, then decode to seconds and snap back to real frame numbers.
# --------------------------------------------------------------------------------------
_EXTRACTOR = {"band": None}
# The fine-tuned backbone, when one is shipped beside the checkpoint.  See
# ``_ft_extractor``: its presence is what selects the fine-tuned feature layout, because
# a 3-band mean-pooled fine-tuned block is the same WIDTH as the frozen one and the two
# cannot be told apart from ``in_dim`` alone.
_FT = {"probed": False, "ext": None, "path": None}
FT_BACKBONE_NAMES = ("stage2_backbone.pt", "s2_backbone.pt")


def _band_extractor(device: str):
    if _EXTRACTOR["band"] is None:
        import stage2_features as S2F
        _EXTRACTOR["band"] = S2F.BandExtractor(device=device, root=_weights_root())
    return _EXTRACTOR["band"]


def _ft_backbone_file():
    """Path of the shipped fine-tuned backbone, or None."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.path.join(here, "model", "stage2"), os.path.join(here, "models", "stage2"),
             os.path.join(os.path.dirname(here), "models", "stage2"),
             os.path.join(os.path.dirname(here), "model", "stage2"), here,
             os.path.join(os.getcwd(), "models", "stage2"),
             os.path.join(os.getcwd(), "model", "stage2")]
    for r in roots:
        for name in FT_BACKBONE_NAMES:
            cand = os.path.join(r, name)
            if os.path.isfile(cand):
                return cand
    return None


def _ft_extractor(device: str):
    """``(FTBandExtractor | None, path)``, built once.

    A failure to build is NOT fatal: the frozen path still works, and a folder scored
    with the frozen features beats a folder scored with the OpenCV heuristic.  But it
    changes the feature layout, so it is logged, and ``_cnn_block`` asks the loaded net
    which layout it was trained for before using either.
    """
    if _FT["probed"]:
        return _FT["ext"], _FT["path"]
    path = _ft_backbone_file()
    _FT["path"] = path
    if path is None:
        _FT["probed"] = True
        return None, None
    # A FAILURE HERE POISONS THE WHOLE RUN, so it gets a second attempt before it latches.
    # `preflight` calls this once at load time; if it returns None, stage2_infer logs
    # PREFLIGHT FAILED and every folder in the run falls back to the OpenCV heuristic --
    # 0.238, not 0.547. Observed once on 2026-08-29 under concurrent GPU/RAM load:
    #   [stage2_model] fine-tuned backbone .../stage2_backbone.pt unusable:
    #   AttributeError: 'NoneType' object has no attribute 'size'
    # and the identical call succeeded on the next five runs. A transient build failure
    # must not be a permanent verdict; a genuine one still latches on the second try, at a
    # cost of one extra 116 MB torch.load.
    for attempt in (1, 2):
        try:
            import stage2_backbone as S2B

            _FT["ext"] = S2B.FTBandExtractor(path, device=device)
            break
        except Exception as exc:                                 # noqa: BLE001
            import sys as _s
            print(f"[stage2_model] fine-tuned backbone {path} unusable "
                  f"(attempt {attempt}/2): {type(exc).__name__}: {exc}",
                  file=_s.stderr, flush=True)
            _FT["ext"] = None
            if attempt == 1:
                try:                      # a transient allocation failure often clears
                    import gc

                    import torch as _t
                    gc.collect()
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception:                                # noqa: BLE001
                    pass
    _FT["probed"] = True
    return _FT["ext"], path


def _weights_root() -> str:
    """A directory in which ``stage2_features.weight_path(root)`` resolves to a real file.

    ``weight_path`` hard-codes ``<root>/models/pretrained/convnext_tiny_in1k.pt``, but the
    submission bundle ships pretrained weights under ``model/stage2/``. Rather than edit
    that contract, find the file wherever it actually is and, when the layout does not
    match, mirror it into a temp dir with a symlink so the expected path exists. Getting
    this wrong is not a slow path -- ``BandExtractor`` falls back to a timm download, which
    on the offline server raises and silently demotes Stage 2 to the heuristic.
    """
    import os
    import tempfile

    import stage2_features as S2F

    name = f"{S2F.BACKBONE}_in1k.pt"
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.dirname(here), os.getcwd()]
    for r in roots:                                   # already-correct layout wins
        if os.path.isfile(os.path.join(r, S2F.WEIGHT_DIR, name)):
            return r
    for r in roots:                                   # bundle layout: model/stageN/<name>
        for sub in ("model/stage2", "model", "models/stage2", "models"):
            cand = os.path.join(r, sub, name)
            if os.path.isfile(cand):
                shim = os.path.join(tempfile.gettempdir(), "s2_weights_root")
                dst = os.path.join(shim, S2F.WEIGHT_DIR)
                os.makedirs(dst, exist_ok=True)
                link = os.path.join(dst, name)
                if not os.path.exists(link):
                    try:
                        os.symlink(cand, link)
                    except OSError:
                        import shutil
                        shutil.copy2(cand, link)
                return shim
    return here


def preflight(device: str = "cpu", net=None) -> str:
    """Build the CNN backbone once, at load time, and return the weight file it used.

    ``predict_folder`` needs ``stage2_features.BandExtractor``.  When the backbone weights
    are not under any root ``_weights_root()`` searches, ``BandExtractor`` falls back to a
    timm download; on the offline server that raises, ``predict_folder`` returns None for
    every folder, and Stage 2 silently runs the OpenCV heuristic.  That is exactly the
    failure that scored 0.23784 instead of 0.42470.  Calling this from ``load_model``
    turns an invisible per-folder demotion into one loud line before any folder is read,
    and costs nothing afterwards because ``_band_extractor`` caches the instance.

    Raises whatever ``BandExtractor`` raises; the caller decides what to do about it.

    ``net`` is the loaded Stage-2 checkpoint, when the caller has it.  It is checked
    against the backbone that is actually on disk, because the two can disagree in a way
    nothing else notices: a fine-tuned net has ``in_dim`` 2382 -- exactly the frozen
    width -- so if ``stage2_backbone.pt`` is missing from the bundle the net loads, runs,
    and answers from ImageNet features it never trained on, with no exception and no log
    line (measured: collision_frame lands on the last sampled frame of most folders).
    That is worse than the OpenCV heuristic, so it is raised here and refused in
    ``_cnn_block``: a loud demotion beats a silent wrong answer.
    """
    import os

    import stage2_features as S2F

    ext, path = _ft_extractor(device)
    if net is not None and bool(getattr(net, "wants_ft", False)) and ext is None:
        meta = dict(getattr(net, "feature_meta", None) or {})
        raise RuntimeError(
            f"the Stage-2 checkpoint was trained on FINE-TUNED features "
            f"(feature_meta backbone={meta.get('backbone')!r}) but no fine-tuned backbone "
            f"file was found -- looked for {list(FT_BACKBONE_NAMES)} beside the checkpoint"
            + (f"; a file was found at {path} but could not be built" if path else "")
            + ". Ship stage2_backbone.pt in model/stage2/ next to best.pt, or ship a "
              "frozen-feature checkpoint. Feeding this net the frozen ImageNet block is a "
              "silent wrong answer on every folder.")
    if ext is not None:
        import sys as _s
        print(f"[stage2_model] fine-tuned backbone: {path} "
              f"(feat_dim {ext.feat_dim}, size {ext.size}, n_frames {ext.n_frames})",
              file=_s.stderr, flush=True)
        return path
    root = _weights_root()
    _band_extractor(device)
    return os.path.join(root, S2F.WEIGHT_DIR, f"{S2F.BACKBONE}_in1k.pt")


def _cnn_block(rgb192, rgb_ft, in_dim: int, device: str, wants_ft: bool):
    """The CNN half of the feature vector, in the layout the loaded net expects.

    ``in_dim`` is the trained net's input width, so the choice is checked against the
    checkpoint rather than assumed:

        ft_dim + flow + fd            fine-tuned backbone only
        ft_dim + 2304 + flow + fd     fine-tuned, with the frozen ImageNet block after it
        2304 + flow + fd              frozen only (also the answer when no backbone shipped)

    A 3-band mean-pooled fine-tuned block happens to be 2304 wide as well, so the last
    two cases are distinguished by whether a backbone file was shipped at all -- shipping
    one means "use it".  When nothing matches, the frozen layout runs and says so; a
    dimension mismatch would otherwise surface as a torch error inside ``predict_folder``
    and demote the folder to the OpenCV heuristic in silence.
    """
    import numpy as np
    import stage2_features as S2F

    rest = S2F.FLOW_DIM + S2F.FD_DIM
    ext, _ = _ft_extractor(device) if wants_ft else (None, None)
    if ext is not None and rgb_ft is not None:
        ft = ext(rgb_ft).astype(np.float32)
        if in_dim == ft.shape[1] + rest:
            return ft
        if in_dim == ft.shape[1] + 2304 + rest:
            return np.concatenate([ft, _band_extractor(device)(rgb192).astype(np.float32)],
                                  axis=1)
        import sys as _s
        print(f"[stage2_model] net in_dim {in_dim} matches no fine-tuned layout "
              f"(ft {ft.shape[1]} + {rest}, or +2304); using the frozen block",
              file=_s.stderr, flush=True)
    elif wants_ft:
        # The net's own feature_meta says fine-tuned, and no fine-tuned backbone could be
        # built.  The frozen block is the same 2304 wide, so returning it here would run to
        # completion and answer nonsense on every folder in silence.  Refuse instead:
        # predict_folder turns this into a logged None and stage2_infer falls back to the
        # heuristic, which is a known 0.238 rather than an unknown wrong answer.
        raise RuntimeError(
            "checkpoint wants fine-tuned features but no fine-tuned backbone is loadable "
            f"(searched {list(FT_BACKBONE_NAMES)} beside the checkpoint); refusing to "
            "substitute the frozen ImageNet block")
    return _band_extractor(device)(rgb192).astype(np.float32)


def _cnn_layout(in_dim: int, device: str) -> tuple:
    """Widths of the CONCATENATED visual sub-blocks ``_cnn_block`` will emit, in order.

    Mirrors that function's three-way decision without running a backbone, because
    ``stage2_side.mirror_features`` has to swap the left and right band of EACH sub-block
    and cannot recover the split from the total width: a stacked ``[ft 2304 | frozen 2304]``
    vector is 4608 wide, divides by three, and mirrors to nonsense in silence.  Mean+max
    pooling is two 3-band groups inside one block, so it is reported as two.
    """
    import stage2_features as S2F

    rest = S2F.FLOW_DIM + S2F.FD_DIM
    ext, _ = _ft_extractor(device)
    if ext is not None:
        ft = int(getattr(ext, "feat_dim", 0))
        pool = getattr(getattr(ext, "net", None), "pool", "mean")
        parts = (ft // 2, ft // 2) if (pool == "meanmax" and ft % 2 == 0) else (ft,)
        if in_dim == ft + rest:
            return parts
        if in_dim == ft + 2304 + rest:
            return parts + (2304,)
    return (int(in_dim) - rest,)


def _frames_to_blocks(frame_paths, want_idx, device: str, in_dim: int = 2382,
                      wants_ft: bool = False):
    """Decode the chosen frames and build (cnn, flow, fd) exactly as training did."""
    import cv2
    import numpy as np
    import stage2_features as S2F
    import s2flow as S3F
    import torch

    cv2.setNumThreads(1)
    T = len(want_idx)
    ext, _ = _ft_extractor(device) if wants_ft else (None, None)
    S = ext.size if ext is not None else 0
    rgb = np.zeros((T, S2F.CNN_SIZE[1], S2F.CNN_SIZE[0], 3), np.uint8)
    rgb_ft = np.zeros((T, S, S, 3), np.uint8) if S else None
    gflow = np.zeros((T, S2F.FLOW_SIZE[1], S2F.FLOW_SIZE[0]), np.uint8)
    ggray = np.zeros((T, S2F.GRAY_SIZE[1], S2F.GRAY_SIZE[0]), np.uint8)
    last = None
    for i, j in enumerate(want_idx):
        img = cv2.imread(str(frame_paths[j]), cv2.IMREAD_COLOR)
        if img is None:
            img = last
            if img is None:
                continue
        last = img
        rgb[i] = cv2.resize(img, S2F.CNN_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1]
        if rgb_ft is not None:
            # from the ORIGINAL frame, not from the 192 px copy: the fine-tuned backbone
            # was trained on a direct full-frame square resize and a double downsample
            # is a different image.
            rgb_ft[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gflow[i] = cv2.resize(g, S2F.FLOW_SIZE, interpolation=cv2.INTER_AREA)
        ggray[i] = cv2.resize(g, S2F.GRAY_SIZE, interpolation=cv2.INTER_AREA)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fh = S2F.FLOW_SIZE[1] // S2F.FLOW_POOL
    fw = S2F.FLOW_SIZE[0] // S2F.FLOW_POOL
    flows = np.zeros((T, 2, fh, fw), np.float32)
    prev = None
    for i in range(T):
        if prev is not None:
            fl = dis.calc(prev, gflow[i], None)
            fl = fl.reshape(fh, S2F.FLOW_POOL, fw, S2F.FLOW_POOL, 2).mean(axis=(1, 3))
            flows[i] = fl.transpose(2, 0, 1)
        prev = gflow[i]
    torch.set_num_threads(1)
    flow_feat = S3F.flow_features(torch.from_numpy(flows)).numpy().astype(np.float32)
    fd = S2F._fd_features(ggray)
    cnn = _cnn_block(rgb, rgb_ft, int(in_dim), device, wants_ft)
    return np.concatenate([cnn, flow_feat, fd], axis=1)


_ENSEMBLE = {"nets": None}


def _feature_key(meta) -> str:
    """A short identity for the CNN block a checkpoint was trained on.

    The fine-tuned 3-band block and the frozen ImageNet one are both 2304 wide, so
    ``in_dim`` cannot separate them, and averaging one kind of net with the other is a
    silent wrong answer on every folder -- worse than either alone and invisible in the
    server log.  Two nets may only be ensembled when this key matches exactly.  A
    checkpoint with no ``feature_meta`` predates the fine-tuned pipeline and is a
    frozen-feature net.
    """
    import os

    m = dict(meta or {})
    bb = m.get("backbone")
    if not bb:
        return "frozen"
    return f"ft:{os.path.basename(str(bb))}:{int(bool(m.get('with_frozen')))}"


#
# 2026-08-31 day4d DIAGNOSTIC. Every submission this project has ever shipped (sub1
# through day4c) has had this ensemble on -- it was never isolated on the leaderboard,
# only on off-leaderboard held-out views (ccd-long, aihub-long, both cited below). Those
# views are reasonably strong evidence (ccd-long is a real-label surface, and the
# structural argument -- a single net answering LEFT on 83% of clips is a macro-F1 0.333
# collapse rule 4 can no longer rebalance away -- does not depend on any one dataset), so
# this is not suspected of being wrong the way the free-space-evasion swap was. It is
# being isolated anyway, now that a submission is finally available and every other open
# Stage 2 axis (entry cap, evasion) is closed for today, to convert "reasonably strong
# off-leaderboard evidence" into an actual leaderboard data point before more work is
# built on top of it.
#
# RESULT + CURRENT STATE (reconciled 2026-09-03). The OFF probe was day4d, S2 0.55287
# against day4c's 0.56364 with the ensemble ON -- so the ensemble is worth about +0.0108,
# a single A/B pair. The default below is back to "1" and the ensemble SHIPS; this comment
# used to still say "testing OFF today", which made the leaderboard record unreadable
# (the file could not tell you which submission carried which). The eval server sets no
# custom env, so the default is what actually ships, not whatever STAGE2_ENSEMBLE_FOLDS
# reads locally. Ensemble ON: submissions 5-7, day4a-c, and every 09-01 build. OFF: day4d only.
ENSEMBLE_FOLDS = os.environ.get("STAGE2_ENSEMBLE_FOLDS", "1") not in ("0", "false", "")


def _ensemble_nets(primary):
    """The primary checkpoint plus any sibling fold checkpoints, loaded once.

    Averaging v2_final with v2_fold0/v2_fold1 was measured better on every held-out view
    (ccd-long 0.7265 vs 0.7179, aihub-long collision 0.7785 vs 0.7721, entry 0.6581 vs
    0.6260) and, more importantly, it un-collapses the two categorical heads: v2_final
    alone answers LEFT for 83% of long clips and evasion=0 for 89%, and a collapsed head
    is macro-F1 0.333 instead of ~0.52. Since inference.py no longer re-balances across
    files (rule 4 forbids it), that collapse would land directly on the score.

    Only the 4.45M-param temporal net re-runs per member; the CNN/flow features are shared,
    so the whole ensemble costs about 0.1 s more per folder.

    A member joins only when its input width AND its feature key match the primary's, so
    a directory holding both frozen-feature and fine-tuned checkpoints ensembles each
    family with itself and says out loud what it skipped.

    Set ``ENSEMBLE_FOLDS = False`` (or ``STAGE2_ENSEMBLE_FOLDS=0``) to ship the primary
    checkpoint alone -- see the day4d comment above for why this is being A/B'd.
    """
    if _ENSEMBLE["nets"] is not None:
        return _ENSEMBLE["nets"]
    import os
    import sys as _s

    import torch

    if not ENSEMBLE_FOLDS:
        print("[stage2_model] ensemble: folds disabled (ENSEMBLE_FOLDS=False), "
              "primary checkpoint only", file=_s.stderr, flush=True)
        _ENSEMBLE["nets"] = [primary]
        return _ENSEMBLE["nets"]

    nets = [primary]
    device = next(primary.parameters()).device
    want_key = _feature_key(getattr(primary, "feature_meta", None))
    want_dim = int(getattr(primary, "in_dim", 0))
    here = os.path.dirname(os.path.abspath(__file__))
    # ``here`` itself comes first for the team package layout, where this module and the
    # fold checkpoints share one directory (submission/model/stage2/).  The three paths
    # after it are the flat submit.zip layout this file shipped in originally, where the
    # modules sit at the archive root and the weights hang under model/stage2/.  Adding a
    # root is additive: the flat layout still resolves exactly as before.
    roots = [here,
             os.path.join(here, "model", "stage2"), os.path.join(here, "models", "stage2"),
             os.path.join(os.path.dirname(here), "models", "stage2")]
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            # The bundle names ensemble members v2_fold*.pt whichever family they belong
            # to; the feature key below is what keeps two families apart.  Matching more
            # names than this would let an unstamped stray checkpoint -- one written
            # before ``feature_meta`` existed, and therefore read as "frozen" -- join a
            # frozen ensemble it was never trained for.
            if not name.startswith("v2_fold") or not name.endswith(".pt") or name in seen:
                continue
            seen.add(name)
            try:
                ck = torch.load(os.path.join(root, name), map_location="cpu",
                                weights_only=False)
                sd = ck.get("model", ck.get("state", ck))
                cfg = dict(ck.get("cfg", {}) or {})
                meta = ck.get("feature_meta") or cfg.get("feature_meta")
                key = _feature_key(meta)
                dim = int(cfg.get("in_dim", ck.get("in_dim", 2382)))
                if key != want_key or dim != want_dim:
                    print(f"[stage2_model] ensemble member {name} SKIPPED: features "
                          f"{key}/{dim} != primary {want_key}/{want_dim}",
                          file=_s.stderr, flush=True)
                    continue
                # Every field must come from the checkpoint: the fold nets use a 7-deep
                # dilation stack and Stage2Net's default is 6, which silently produced an
                # "unexpected key blocks.6.*" load error and a one-member "ensemble".
                kw = {k: cfg[k] for k in ("hidden", "dilations", "dropout",
                                          "feature_meta") if k in cfg}
                m = Stage2Net(dim, **kw)
                m.load_state_dict(sd)
                nets.append(m.eval().to(device))
            except Exception as exc:                     # noqa: BLE001
                print(f"[stage2_model] ensemble member {name} skipped: "
                      f"{type(exc).__name__}: {exc}", file=_s.stderr, flush=True)
        if len(nets) > 1:
            break
    print(f"[stage2_model] ensemble: {len(nets)} member(s) [{want_key}]",
          file=_s.stderr, flush=True)
    _ENSEMBLE["nets"] = nets
    return nets


def predict_folder(net, folder, frames=None, frame_numbers=None, mode: str = "full",
                   sec_per_item: float | None = None) -> dict:
    """Run the trained Stage 2 network over one frame folder.

    Returns the four submission fields. Raises nothing that stage2_infer cannot handle:
    on any failure it returns ``None`` so the caller falls back to its heuristic.
    """
    try:
        import numpy as np
        import torch
        import s2common as common
        import stage2_features as S2F

        paths = list(frames or common.list_frames(folder))
        if len(paths) < 4:
            return None
        numbers = [int(x) for x in (frame_numbers or [common.frame_number(p) for p in paths])]
        n = len(paths)
        fps = common.s2_assumed_fps(n)
        duration = n / max(fps, 1e-6)

        # Sample on the training grid (HZ per second), capped so a 1250-frame folder costs
        # a bounded amount. 'min' is the degraded path stage2_infer picks under time pressure.
        cap = {"min": 96, "fast": 192}.get(mode, 384)
        n_take = int(min(cap, max(8, round(duration * S2F.HZ))))
        want = np.unique(np.rint(np.linspace(0, n - 1, n_take)).astype(int))
        t_sec = np.asarray([numbers[j] for j in want], dtype=np.float64) / max(fps, 1e-6)
        t_sec = t_sec - t_sec[0]

        model = net["model"] if isinstance(net, dict) and "model" in net else net
        device = next(model.parameters()).device
        # The backbone must sit on the same device as the net; stage2_infer hands us the
        # bare module, so the device comes from the module, not from a handle dict.
        wants_ft = bool(getattr(model, "wants_ft", False))
        feats = _frames_to_blocks(paths, want, str(device),
                                  int(getattr(model, "in_dim", 2382)), wants_ft)
        x = torch.from_numpy(feats[None]).to(device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
        members = _ensemble_nets(model)
        acc = None
        with torch.no_grad():
            for m in members:
                o = m(x, mask)
                cur = {k: o[k].float().cpu().numpy() for k in
                       ("collision", "entry", "side", "evasion")}
                acc = cur if acc is None else {k: acc[k] + cur[k] for k in acc}
        o = {k: v / len(members) for k, v in acc.items()}
        cl, el = o["collision"], o["entry"]
        cs, es = decode(cl, el, [len(want)], [t_sec], smooth_sigma=0.0, mode="window",
                        entry_max_lead_sec=_entry_max_lead_default())
        side = SIDE_LABELS[int(o["side"].argmax(1)[0])]
        evasion = int(o["evasion"].argmax(1)[0])

        # seconds -> the nearest frame number that actually exists on disk
        def to_frame(sec: float) -> int:
            k = int(np.argmin(np.abs(t_sec - float(sec))))
            return int(numbers[want[k]])

        c_frame = to_frame(cs[0])
        e_frame = to_frame(es[0])
        if e_frame > c_frame:
            e_frame = c_frame

        # ---- LEVER A hook (entry_frame) -----------------------------------------------
        # stage2_entry owns the entry column.  `decode` above already bounded the SEARCH;
        # this bounds the ANSWER in the folder's own frame numbering (the two agree unless
        # the sample grid is coarse) and is also the switch that turns the submission into
        # a leaderboard probe of the private lead distribution.  Pure integer arithmetic,
        # no I/O, no model: it cannot cost time and it cannot raise.
        try:
            import stage2_entry

            e_frame = int(stage2_entry.apply_entry_policy(
                c_frame, e_frame, fps, frame_numbers=[numbers[j] for j in want]))
        except Exception:                                       # noqa: BLE001
            pass
        # ---- end LEVER A hook ---------------------------------------------------------

        # ---- LEVER 4 hook -------------------------------------------------------------
        # stage2_side owns entry_side / evasion_space.  It re-runs the SAME members on
        # this clip's own features and on their free mirror (a permutation of the 2382-d
        # vector); the mirror cancels the constant "say LEFT" bias that collapsed the side
        # head off-domain, and a fixed logit bias re-centres evasion_space.  Costs one
        # extra forward pass per member (~0.03 s) plus ~30 half-resolution jpeg reads
        # (~0.15 s), touches nothing else, and returns None on any failure so the two
        # argmax answers above stand.
        try:
            import stage2_side
            _ov = stage2_side.override(members, x, mask, frame_paths=paths,
                                       frame_numbers=numbers, fps=fps,
                                       collision_frame=c_frame,
                                       cnn_blocks=_cnn_layout(
                                           int(getattr(model, "in_dim", 2382)), str(device)),
                                       wants_ft=wants_ft)
            if _ov:
                side, evasion = _ov["entry_side"], _ov["evasion_space"]
        except Exception:                                   # noqa: BLE001
            pass
        # ---- end LEVER 4 hook ---------------------------------------------------------
        return {"collision_frame": c_frame, "entry_frame": e_frame,
                "entry_side": side, "evasion_space": evasion}
    except Exception as exc:                                    # noqa: BLE001
        # stage2_infer treats None as "use the heuristic", but a silent None is how the
        # first submission shipped a heuristic while believing it shipped a model. Say why.
        import os
        import sys as _sys
        import traceback

        print(f"[stage2_model] predict_folder failed: {type(exc).__name__}: {exc}",
              file=_sys.stderr, flush=True)
        if os.environ.get("STAGE2_DEBUG"):
            traceback.print_exc()
        return None


__all__ = __all__ + ["predict_folder", "preflight"]
