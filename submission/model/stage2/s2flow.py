"""Stage 3 optical-flow features.

Stage 3 asks for an (accel, steer) label every 0.1 s. Both are properties of the
*ego* motion, and the only observable is the dashcam image, so the feature front-end
turns each 0.1 s interval into the geometry of its optical-flow field:

  * how fast the world streams past  -> |flow| quantiles per horizon-referenced row band
  * where the flow field expands from -> focus of expansion + horizon-band mean u (yaw)
  * how strongly it expands           -> divergence                          (speed/depth)
  * how much of the image stands still-> still-pixel fractions               (STOPPED gate)
  * where the motion sits in frame    -> 3x3 grid of |flow|, u and v

The reduction is deliberately *interpretable* and low-dimensional (70 numbers per
sample) so a small temporal CNN can be trained on a few hundred clips without
overfitting, and so the same features can be produced by RAFT on the GPU or by DIS on
the CPU when the time budget runs short -- the two are interchangeable because they share
``flow_features``.

Decoding note (measured on this machine, 2026-08-26): PyAV's *frame*-threaded decode
(``thread_type="AUTO"``, the default of ``common.decode_frames``) deadlocks -- every
thread parked in futex, SIGTERM ignored -- on the second and later ``av.open`` inside a
process that has already built a CUDA module; it reproduced in roughly half of the runs.
So this module decodes with ``cv2.VideoCapture`` (a separate FFmpeg build, 1.5 ms/frame
vs 2.5-6.8 ms for the safe PyAV modes, and it reads comma2k19's raw ``video.hevc``
elementary streams exactly as well), and only falls back to ``common.decode_frames`` with
slice threading, which does not deadlock but is ~4x slower.

Runtime contract (used by inference.py)::

    meta = common.probe_video(path, fallback_fps=common.S3_FALLBACK_FPS)
    n, fps = sample_grid(meta)                          # one row per submitted sample
    ex = FlowExtractor(device="cuda", model="raft_large", size=(512, 384))
    feats = ex.extract(path, n, fps)                    # [n, FEATURE_DIM] float32

Nothing here imports the training-time label code.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as common  # noqa: E402  (flat module layout: inference.py sits next to these)

# --------------------------------------------------------------------------------------
# Geometry of the reduction. All ROIs are fractions of (H, W) so the features do not
# depend on the source resolution.
#
# Why a row-band *profile* instead of one road ROI: on a flat road the flow of a ground
# point is |flow| = (y - y_horizon)^2 * V * dt / (f * h), i.e. it grows quadratically
# towards the bottom of the frame. A single ROI average therefore mixes 3 px and 80 px
# pixels, and -- worse -- on the night clips of this competition the road surface has no
# texture at all, so the average is dominated by pixels where the flow estimate is simply
# zero. Measured on the five public Stage 3 clips, mean |flow| over a 0.55-1.0 ROI
# correlates r=+0.07 with CAN speed, while the p90 of a single 0.03-tall band just under
# the horizon reaches r=+0.75. So: bands, and high quantiles inside each band.
# --------------------------------------------------------------------------------------
BAND_Y0, BAND_Y1, N_BANDS = 0.45, 1.00, 8   # row bands from just above the horizon down
BAND_X = (0.10, 0.90)
ROAD_ROI = (0.50, 0.90, 0.10, 0.90)      # ground plane, hood excluded
SKY_ROI = (0.10, 0.42, 0.05, 0.95)       # above the horizon: other traffic / rotation only
HORIZON_ROI = (0.38, 0.55, 0.05, 0.95)   # band around the vanishing point -> nearly pure rotation
STILL_PX = 0.5                            # |flow| below this counts as "not moving"
FOE_MIN_MAG = 0.5                         # pixels quieter than this carry no direction information

# PyAV thread mode for the fallback decode path. "AUTO" (frame threading) deadlocks in a
# process that has already built a CUDA module -- see the module docstring -- so the
# fallback uses slice threading, which is ~4x slower but never wedges.
DECODE_THREADS = "SLICE"

_GRID = 3


def _grid_names(prefix: str) -> list[str]:
    return [f"{prefix}_r{r}c{c}" for r in range(_GRID) for c in range(_GRID)]


FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"band_p50_{i}" for i in range(N_BANDS)]
    + [f"band_p90_{i}" for i in range(N_BANDS)]
    + [
        "sky_p50", "sky_p90",
        "mag_mean_road", "mag_p25_road", "mag_med_road", "mag_p75_road", "mag_p90_road",
        "horiz_mean_u", "horiz_std_u", "horiz_mean_v", "global_mean_v",
        "foe_x", "foe_y", "foe_resid",
        "div_road", "curl_road",
        "still_frac_road", "still_frac_off",
    ]
    + _grid_names("grid_mag_mean")
    + _grid_names("grid_mag_std")
    + _grid_names("grid_u_mean")
    + _grid_names("grid_v_mean")
)
FEATURE_DIM = len(FEATURE_NAMES)

# Features whose sign flips under a horizontal mirror of the image. Training-time
# augmentation needs this and so does any test-time-flip ensembling.
_FLIP_SIGN_NAMES = {"horiz_mean_u", "foe_x", "curl_road"}
_FLIP_SIGN_NAMES |= set(_grid_names("grid_u_mean"))
# Features that swap places under a horizontal mirror (column 0 <-> column 2).
_FLIP_SWAP_PREFIXES = ("grid_mag_mean", "grid_mag_std", "grid_u_mean", "grid_v_mean")


def _flip_plan() -> tuple[np.ndarray, np.ndarray]:
    """(permutation, sign) arrays implementing a horizontal mirror in feature space."""
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    perm = np.arange(FEATURE_DIM, dtype=np.int64)
    sign = np.ones(FEATURE_DIM, dtype=np.float32)
    for n, i in idx.items():
        if n in _FLIP_SIGN_NAMES:
            sign[i] = -1.0
        for p in _FLIP_SWAP_PREFIXES:
            if n.startswith(p + "_r"):
                r, c = int(n[-3]), int(n[-1])
                perm[i] = idx[f"{p}_r{r}c{_GRID - 1 - c}"]
                break
    return perm, sign


FLIP_PERM, FLIP_SIGN = _flip_plan()


def flip_features(x: np.ndarray) -> np.ndarray:
    """Mirror a feature matrix ``[..., D]`` as if the video had been flipped horizontally."""
    return (x[..., FLIP_PERM] * FLIP_SIGN).astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------
# Decoding and the sample grid.
# --------------------------------------------------------------------------------------
# Every extractor resizes to a fixed 4:3 working size (384x288 for DIS, 512x384 for RAFT).
# A source that is not 4:3 is therefore squashed *anisotropically*: a 1280x720 dashcam frame
# going to 384x288 shrinks x by 0.300 and y by 0.400, so every horizontal displacement ends
# up 25% smaller than the matching vertical one. comma2k19 -- the whole Stage 3 training
# corpus -- is 1164x874, 4:3 to within one row, so nothing in training contains that
# distortion and stage3_aug.rescale_flow cannot represent it (a single global factor).
#
# ``fit_aspect`` below centre-crops the incoming frame to the working ratio, which makes the
# resize isotropic again. It is DISABLED by default, and the reason is measured, not
# theoretical. Cropping a 16:9 frame back to 4:3 buys isotropy at the price of a 1.33x global
# flow magnification (the same 384 px now spans 75% of the source width), and the two effects
# land on different halves of the score: isotropy helps *steer*, magnification hurts *accel*,
# and accel carries 0.7 of the Stage 3 weight. Scored on the organizers' own public five
# rendered 16:9 (ar169 -> ar169_fix), dense key, 2026-08-29:
#
#   model scoring the public five            d(accel)  d(steer)  d(total)
#   final_* pair + submission-4 bias          -0.033    +0.058    -0.0056
#   LOVO pair + submission-4 bias             -0.023    +0.069    +0.0044
#   LOVO pair + leverD bias                   -0.019    +0.070    +0.0079
#   the CURRENTLY SHIPPED 6-net + leverD bias -0.029    +0.010    -0.0177
#
# accel loses under every model; the total only survives when the steer gain is large, and on
# the bundle that actually ships the steer gain is gone (leverD's nets and bias already fixed
# steer) so only the accel loss is left. An independent check -- the same five clips
# re-rendered as a real 1280x720 @ 10 fps dashcam and pushed through stage3_infer.predict_video
# with the shipped bundle -- agrees: pub50 0.5632 -> 0.5319, dense 0.5973 -> 0.5773.
#
# On a 4:3 source the crop is a bit-exact no-op (verified end to end: 0/1198 rows changed on
# OPEN_001+OPEN_003), so leaving it off costs nothing there either, and the Stage 3 public
# examples are themselves 1164x874 comma2k19 segments -- the evidence that the private clips
# are 16:9 at all is indirect (the Stage 1/2 example footage is 1280x720). Flip MATCH_ASPECT
# back to True only with a measurement on the shipped bundle that says the total improves.
MATCH_ASPECT = False
# A source within this fraction of the working aspect ratio is left alone even when
# MATCH_ASPECT is on. comma2k19 is 1164x874 = 1.3318 against the 1.3333 of every working
# size -- 0.11% out -- and cropping it to exactly 4:3 drops one row, which is harmless in
# principle but not free in practice: re-sampling 873 rows instead of 874 into 288 moves the
# cached features by ~5% of their own scale. The tolerance keeps the training domain
# bit-identical while still catching the 33% error of a 16:9 dashcam.
ASPECT_TOL = 0.02


def fit_aspect(img, size: tuple[int, int]):
    """Centre-crop ``img`` (HxWx3) to the aspect ratio of ``size`` = (W, H).

    A no-op unless the source is more than ``ASPECT_TOL`` away from that ratio.
    """
    if not MATCH_ASPECT:
        return img
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    ar = float(size[0]) / float(size[1])
    if abs((w / h) / ar - 1.0) <= ASPECT_TOL:
        return img
    nh = int(round(w / ar))
    if nh < h:
        o = (h - nh) // 2
        return img[o:o + nh]
    nw = int(round(h * ar))
    if nw < w:
        o = (w - nw) // 2
        return img[:, o:o + nw]
    return img


def _iter_frames_cv2(path, indices: Sequence[int], size: tuple[int, int]):
    """Yield ``(index, RGB uint8 HxWx3)`` for the requested indices using OpenCV.

    ``grab()`` skips the colour-convert/resize for frames nobody asked for, which is most
    of them when the sample grid is coarser than the frame rate.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return
        want = list(indices)
        wi, i, last = 0, 0, None
        while wi < len(want):
            if not cap.grab():
                break
            if i == want[wi]:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    last = cv2.cvtColor(
                        cv2.resize(fit_aspect(bgr, size), size, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)
                arr = last
                while wi < len(want) and want[wi] == i:
                    if arr is not None:
                        yield want[wi], arr
                    wi += 1
            i += 1
    finally:
        cap.release()


def sample_grid(meta, n_samples: int | None = None) -> tuple[int, float]:
    """(n_samples, fps) that tiles exactly the frames the file actually contains.

    ``common.s3_sample_count`` is the authority on how many rows a submission needs
    (one per decoded frame for the private 10 fps clips). The pairing step must then be
    ``n_frames / n_samples`` frames, which is what an effective fps of
    ``10 * n_frames / n_samples`` produces. Training clips from comma2k19 are 20 fps with
    labels on a true 0.1 s grid: pass ``n_samples = round(duration * 10)`` and the same
    formula returns 20.0.
    """
    n = int(n_samples) if n_samples else int(common.s3_sample_count(meta))
    n = max(1, n)
    n_fr = max(1, int(meta.n_frames))
    return n, float(common.S3_HZ) * n_fr / n


# --------------------------------------------------------------------------------------
# Flow -> features. One implementation, shared by RAFT and DIS, so the two extractors
# are interchangeable at inference time.
# --------------------------------------------------------------------------------------
def flow_features(flow):
    """``flow`` is a torch tensor ``[B, 2, H, W]`` in pixels; returns ``[B, D]`` float32.

    Every output is finally squashed with ``sign(x) * log1p(|x|)``: flow magnitudes span
    0.1 px (a stopped car) to 100 px (a highway bottom row) and a linear scale would make
    the normalisation of the model meaningless.
    """
    import torch
    import torch.nn.functional as F

    flow = flow.float()
    B, _, H, W = flow.shape
    u, v = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(u * u + v * v + 1e-12)
    dev = flow.device
    q_bands = torch.tensor([0.5, 0.9], device=dev)

    def _slice(t, roi):
        y0, y1, x0, x1 = roi
        a, b = int(y0 * H), max(int(y0 * H) + 1, int(y1 * H))
        c, d = int(x0 * W), max(int(x0 * W) + 1, int(x1 * W))
        return t[:, a:b, c:d]

    xa, xb = int(BAND_X[0] * W), max(int(BAND_X[0] * W) + 1, int(BAND_X[1] * W))
    # Equal-height bands so all N_BANDS quantiles are one kernel launch instead of eight;
    # ten separate torch.quantile calls cost more than the whole RAFT forward pass.
    bh = max(1, int((BAND_Y1 - BAND_Y0) * H) // N_BANDS)
    y0b = int(BAND_Y0 * H)
    y0b = min(y0b, max(0, H - bh * N_BANDS))
    bands = mag[:, y0b:y0b + bh * N_BANDS, xa:xb].reshape(B * N_BANDS, -1)
    qb = torch.quantile(bands, q_bands, dim=1).reshape(2, B, N_BANDS)
    out = [qb[0], qb[1]]

    sky = _slice(mag, SKY_ROI).reshape(B, -1)
    road = _slice(mag, ROAD_ROI).reshape(B, -1)
    qs = torch.quantile(sky, q_bands, dim=1)
    qr = torch.quantile(road, torch.tensor([0.25, 0.5, 0.75, 0.9], device=dev), dim=1)
    out += [qs[0][:, None], qs[1][:, None]]
    out += [road.mean(1, keepdim=True), qr[0][:, None], qr[1][:, None], qr[2][:, None], qr[3][:, None]]

    hu = _slice(u, HORIZON_ROI).reshape(B, -1)
    hv = _slice(v, HORIZON_ROI).reshape(B, -1)
    hu_med = hu.median(dim=1, keepdim=True).values
    # MAD is the right spread estimate here: a single passing truck must not swamp the
    # yaw proxy the way a plain std would.
    hu_mad = (hu - hu_med).abs().median(dim=1, keepdim=True).values * 1.4826
    out += [hu.mean(1, keepdim=True), hu_mad, hv.mean(1, keepdim=True),
            v.reshape(B, -1).mean(1, keepdim=True)]

    # Focus of expansion: every flow vector defines a line through its own pixel; under
    # pure translation all those lines meet at the FoE. Solve the 2x2 weighted
    # least-squares in normalised image coordinates so the answer is resolution-free.
    ys = (torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2) / (H / 2)
    xs = (torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2) / (W / 2)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    Un, Vn = u / (W / 2), v / (H / 2)
    w = torch.where(mag > FOE_MIN_MAG, mag, torch.zeros_like(mag))
    a1, a2 = -Vn, Un
    b = -Vn * X + Un * Y
    s11 = (w * a1 * a1).sum((1, 2))
    s12 = (w * a1 * a2).sum((1, 2))
    s22 = (w * a2 * a2).sum((1, 2))
    r1 = (w * a1 * b).sum((1, 2))
    r2 = (w * a2 * b).sum((1, 2))
    wsum = w.sum((1, 2)).clamp_min(1e-6)
    reg = 1e-3 * wsum
    det = (s11 + reg) * (s22 + reg) - s12 * s12
    det = torch.where(det.abs() < 1e-9, torch.full_like(det, 1e-9), det)
    fx = (((s22 + reg) * r1 - s12 * r2) / det).clamp(-4.0, 4.0)
    fy = (((s11 + reg) * r2 - s12 * r1) / det).clamp(-4.0, 4.0)
    resid = ((w * (a1 * fx[:, None, None] + a2 * fy[:, None, None] - b) ** 2).sum((1, 2)) / wsum).sqrt()
    out += [fx[:, None], fy[:, None], resid[:, None]]

    du_dy, du_dx = torch.gradient(u, dim=(1, 2))
    dv_dy, dv_dx = torch.gradient(v, dim=(1, 2))
    out += [_slice(du_dx + dv_dy, ROAD_ROI).reshape(B, -1).mean(1, keepdim=True),
            _slice(dv_dx - du_dy, ROAD_ROI).reshape(B, -1).mean(1, keepdim=True)]

    still = (mag < STILL_PX).float()
    road_mask = torch.zeros(1, H, W, device=dev)
    y0, y1, x0, x1 = ROAD_ROI
    road_mask[:, int(y0 * H):max(int(y0 * H) + 1, int(y1 * H)),
              int(x0 * W):max(int(x0 * W) + 1, int(x1 * W))] = 1.0
    rm = road_mask.expand(B, -1, -1)
    n_road = rm.sum((1, 2)).clamp_min(1.0)
    n_off = (1 - rm).sum((1, 2)).clamp_min(1.0)
    # Two separate gates: a stopped ego car still sees other traffic move, so the
    # off-road (upper/side) fraction alone would be misleading.
    out += [((still * rm).sum((1, 2)) / n_road)[:, None],
            ((still * (1 - rm)).sum((1, 2)) / n_off)[:, None]]

    g_mag = F.adaptive_avg_pool2d(mag.unsqueeze(1), (_GRID, _GRID)).reshape(B, -1)
    g_mag2 = F.adaptive_avg_pool2d((mag * mag).unsqueeze(1), (_GRID, _GRID)).reshape(B, -1)
    g_std = (g_mag2 - g_mag * g_mag).clamp_min(0).sqrt()
    g_u = F.adaptive_avg_pool2d(u.unsqueeze(1), (_GRID, _GRID)).reshape(B, -1)
    g_v = F.adaptive_avg_pool2d(v.unsqueeze(1), (_GRID, _GRID)).reshape(B, -1)
    out += [g_mag, g_std, g_u, g_v]

    feat = torch.nan_to_num(torch.cat(out, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sign(feat) * torch.log1p(feat.abs())


# --------------------------------------------------------------------------------------
# Weight discovery. The training machine may download once; the eval server never can.
# --------------------------------------------------------------------------------------
_RAFT_FILES = {
    "raft_small": "raft_small_C_T_V2-01064c6d.pth",
    "raft_large": "raft_large_C_T_SKHT_V2-ff5fadd5.pth",
}


def default_weight_path(model: str) -> str | None:
    fn = _RAFT_FILES.get(model)
    if not fn:
        return None
    here = Path(__file__).resolve().parent
    cands = []
    env = os.environ.get("STAGE3_RAFT_DIR")
    if env:
        cands.append(Path(env))
    for base in (here, here.parent):
        cands += [base / "models" / "pretrained", base / "models", base / "weights", base]
    for c in cands:
        p = c / fn
        if p.is_file():
            return str(p)
    return None


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------
class _BaseFlowExtractor:
    """Shared frame pairing / batching / decoding logic.

    Subclasses only provide ``_flow(imgs0, imgs1) -> np.ndarray [B, FEATURE_DIM]``:
    they estimate the flow field however they like and hand it to ``flow_features``, so
    RAFT and DIS produce byte-comparable features and can be swapped at inference time.
    """

    name = "base"

    def __init__(self, size: tuple[int, int] = (384, 288), batch_size: int = 8,
                 device: str = "cpu", backend: str = "auto", feat_device: str | None = None):
        self.size = (int(size[0]), int(size[1]))
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.backend = backend
        # The reduction is ~0.12 ms/pair on a GPU and 11-140 ms/pair on CPU threads
        # (the FoE least-squares and torch.gradient touch a dozen full-resolution
        # temporaries), so send it to the GPU whenever there is one -- even for DIS,
        # whose flow itself is computed on the CPU.
        if feat_device is None:
            try:
                import torch

                feat_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                feat_device = "cpu"
        self.feat_device = feat_device

    def _frames(self, path, uniq: Sequence[int]):
        """cv2 first, PyAV (slice-threaded) as the fallback -- see the module docstring."""
        if self.backend in ("auto", "cv2"):
            seen = False
            for item in _iter_frames_cv2(path, uniq, self.size):
                seen = True
                yield item
            if seen or self.backend == "cv2":
                return
        # common.decode_frames resizes inside libswscale, which would re-introduce exactly
        # the anisotropic squash the cv2 path removes -- and the frame reaches this loop
        # already at self.size, so cropping it here would be too late. Read the stream
        # geometry from the container header (no demux, no decode) and only when a crop is
        # actually due, decode at the native size and do the crop + resize here. A 4:3
        # source therefore keeps the old byte-identical fast path.
        native = None
        if MATCH_ASPECT:
            try:
                import av

                with av.open(str(path)) as c:
                    st = c.streams.video[0]
                    w, h = int(st.width or 0), int(st.height or 0)
                if w > 0 and h > 0 and abs(
                        (w / h) / (self.size[0] / self.size[1]) - 1.0) > ASPECT_TOL:
                    native = (w, h)
            except Exception:
                native = None
        if native is None:
            yield from common.decode_frames(path, uniq, size=self.size,
                                            threads=DECODE_THREADS)
            return
        import cv2

        for i, arr in common.decode_frames(path, uniq, threads=DECODE_THREADS):
            yield i, cv2.resize(fit_aspect(arr, self.size), self.size,
                                interpolation=cv2.INTER_AREA)

    # -- to be implemented -------------------------------------------------------------
    def _flow(self, imgs0: np.ndarray, imgs1: np.ndarray):
        raise NotImplementedError

    # -- public ------------------------------------------------------------------------
    def extract(self, video_path, n_samples: int, fps: float,
                hz: float = common.S3_HZ) -> np.ndarray:
        """Feature matrix for ``n_samples`` samples spaced 1/hz seconds apart.

        Sample ``k`` is the flow between the frames that straddle it: indices
        ``round(k*fps/hz)`` and ``round((k+1)*fps/hz)``. Short videos, undecodable
        frames and ``n_samples`` beyond the end of the file all fall back to the last
        good frame instead of shortening the output.
        """
        n_samples = max(1, int(n_samples))
        feats = np.zeros((n_samples, FEATURE_DIM), dtype=np.float32)
        fps = float(fps) if fps and fps > 0 else common.S3_FALLBACK_FPS
        idx = np.rint(np.arange(n_samples + 1) * fps / float(hz)).astype(np.int64)
        idx = np.maximum(idx, 0)

        need: dict[int, int] = defaultdict(int)
        for k in range(n_samples):
            need[int(idx[k])] += 1
            need[int(idx[k + 1])] += 1
        uniq = sorted(need.keys())

        buf: dict[int, np.ndarray] = {}
        b0: list[np.ndarray] = []
        b1: list[np.ndarray] = []
        bk: list[int] = []
        pend = 0
        last = None

        def flush() -> None:
            if not bk:
                return
            try:
                f = self._flow(np.stack(b0), np.stack(b1))
                feats[np.asarray(bk)] = f
            except Exception:
                pass  # a failed batch keeps its zero rows rather than killing the run
            b0.clear(); b1.clear(); bk.clear()

        try:
            for i, arr in self._frames(video_path, uniq):
                if arr is None:
                    continue
                buf[i] = arr
                last = arr
                while pend < n_samples:
                    i0, i1 = int(idx[pend]), int(idx[pend + 1])
                    if i0 not in buf or i1 not in buf:
                        break
                    b0.append(buf[i0]); b1.append(buf[i1]); bk.append(pend)
                    need[i0] -= 1
                    need[i1] -= 1
                    for j in (i0, i1):
                        if need[j] <= 0:
                            buf.pop(j, None)
                    pend += 1
                    if len(bk) >= self.batch_size:
                        flush()
        except Exception:
            pass
        flush()

        # Tail: the video ended before the requested sample grid did.
        if pend < n_samples and last is not None:
            while pend < n_samples:
                i0, i1 = int(idx[pend]), int(idx[pend + 1])
                b0.append(buf.get(i0, last)); b1.append(buf.get(i1, last)); bk.append(pend)
                pend += 1
                if len(bk) >= self.batch_size:
                    flush()
            flush()
        return feats

    # -- helpers -----------------------------------------------------------------------
    def _to_features(self, flow) -> np.ndarray:
        if self.feat_device and flow.device.type != self.feat_device:
            flow = flow.to(self.feat_device, non_blocking=True)
        return flow_features(flow).detach().float().cpu().numpy().astype(np.float32)


class FlowExtractor(_BaseFlowExtractor):
    """RAFT-based extractor.

    ``raft_large`` (C_T_SKHT weights, i.e. fine-tuned on KITTI/Sintel driving data) is
    much the better choice here: on the five public night clips the best flow feature
    correlates r=+0.78 with CAN speed at 512x384 and r=+0.70 at 384x288, against r=+0.29
    for ``raft_small`` (Chairs+Things only), which mostly returns ~0 flow on dark
    low-texture asphalt.
    """

    def __init__(self, device: str = "cuda", model: str = "raft_small",
                 size: tuple[int, int] = (384, 288), amp: bool = True,
                 weights_path: str | None = None, batch_size: int = 8,
                 allow_download: bool = True, backend: str = "auto",
                 iters: int = 12, feat_device: str | None = None):
        import torch
        from torchvision.models.optical_flow import raft_large, raft_small

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        super().__init__(size=size, batch_size=batch_size, device=device, backend=backend,
                         feat_device=feat_device or device)
        self.name = model
        self.amp = bool(amp) and device == "cuda"
        # RAFT downsamples by 8 three times; pad the working size up to a multiple of 8.
        w = (self.size[0] + 7) // 8 * 8
        h = (self.size[1] + 7) // 8 * 8
        self.size = (w, h)

        ctor = raft_small if "small" in model else raft_large
        wp = weights_path or default_weight_path(model)
        if wp and os.path.isfile(wp):
            net = ctor(weights=None, progress=False)
            sd = torch.load(wp, map_location="cpu", weights_only=True)
            net.load_state_dict(sd)
            self.weights_path = wp
        elif allow_download:
            net = ctor(weights="DEFAULT", progress=False)  # training machine only
            self.weights_path = "torchvision:DEFAULT"
        else:
            raise FileNotFoundError(f"RAFT weights for {model} not found offline")
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.iters = max(1, int(iters))

    def _flow(self, imgs0: np.ndarray, imgs1: np.ndarray):
        import torch

        def prep(a: np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
            t = t.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            return t

        with torch.inference_mode():
            x0, x1 = prep(imgs0), prep(imgs1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                flow = self.net(x0, x1, num_flow_updates=self.iters)[-1]
            return self._to_features(flow)


class DISFlowExtractor(_BaseFlowExtractor):
    """OpenCV DIS optical flow -- the GPU-free path, same feature contract as RAFT.

    Measured at 384x288 on this machine: 2.7 ms/pair for ``calc`` plus 1.4 ms/pair for the
    feature reduction, i.e. ~2.5 s for a 600-sample clip -- comparable to raft_large and
    with a *better* speed correlation (r=+0.80 vs +0.78) on the night clips, but it is
    CPU-bound and collapses under thread oversubscription, so pass ``cv_threads`` on a
    small box (the eval server has 7 vCPUs).
    """

    def __init__(self, device: str = "cpu", size: tuple[int, int] = (384, 288),
                 preset: int | None = None, batch_size: int = 8, backend: str = "auto",
                 cv_threads: int | None = None, feat_device: str | None = None,
                 **_ignored):
        import cv2

        if cv_threads:
            cv2.setNumThreads(int(cv_threads))
        super().__init__(size=size, batch_size=batch_size, device="cpu", backend=backend,
                         feat_device=feat_device)
        self.name = "dis"
        p = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if preset is None else preset
        self.dis = cv2.DISOpticalFlow_create(p)
        self.dis.setUseSpatialPropagation(True)

    def _flow(self, imgs0: np.ndarray, imgs1: np.ndarray):
        import cv2
        import torch

        flows = np.empty((len(imgs0), imgs0.shape[1], imgs0.shape[2], 2), dtype=np.float32)
        for i in range(len(imgs0)):
            g0 = cv2.cvtColor(imgs0[i], cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(imgs1[i], cv2.COLOR_RGB2GRAY)
            flows[i] = self.dis.calc(g0, g1, None)
        t = torch.from_numpy(flows).permute(0, 3, 1, 2)
        return self._to_features(t)


def build_extractor(kind: str = "raft_large", device: str = "cuda",
                    size: tuple[int, int] = (384, 288), **kw) -> _BaseFlowExtractor:
    """``kind`` in {"dis", "raft_small", "raft_large"}.

    Never raises: missing weights, no CUDA or an unknown name all degrade to DIS, which
    needs nothing but OpenCV. inference.py relies on that -- an exception there costs the
    whole submission.
    """
    k = (kind or "").lower()
    if k.startswith("raft"):
        try:
            # FlowExtractor has no **kwargs, so a DIS-only argument (cv_threads) would raise
            # TypeError and silently demote the caller to DIS. Filter per constructor.
            return FlowExtractor(device=device, model=k, size=size,
                                 **{a: b for a, b in kw.items()
                                    if a in ("amp", "weights_path", "batch_size",
                                             "allow_download", "backend", "iters",
                                             "feat_device")})
        except Exception as exc:
            print(f"[stage3_features] {k} unavailable ({type(exc).__name__}: {exc}) -> DIS",
                  file=sys.stderr, flush=True)
    return DISFlowExtractor(size=size, **{a: b for a, b in kw.items()
                                          if a in ("preset", "batch_size", "backend",
                                                   "cv_threads", "feat_device")})


__all__ = [
    "FEATURE_NAMES", "FEATURE_DIM", "FLIP_PERM", "FLIP_SIGN", "flip_features",
    "sample_grid", "flow_features", "FlowExtractor", "DISFlowExtractor", "build_extractor",
    "fit_aspect", "MATCH_ASPECT", "ASPECT_TOL",
    "default_weight_path", "ROAD_ROI", "HORIZON_ROI",
]
