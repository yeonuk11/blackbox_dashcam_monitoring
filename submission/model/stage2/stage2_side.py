"""Per-file ``entry_side`` and ``evasion_space`` for Stage 2 (LEVER 4).

WHY THIS FILE EXISTS
--------------------
``entry_side`` and ``evasion_space`` are 15% of Stage 2 each, and both are Macro-F1
over two classes.  That metric has a floor and a ceiling that a plain accuracy number
hides:

* an answer that collapses to one class scores ``p/(1+p) <= 0.5`` -- 0.333 when the
  truth is balanced -- however confident it is;
* an answer that is *pure noise* but whose class MARGINAL matches the truth's scores
  exactly 0.5, for free;
* an answer that is balanced AND right a fraction ``a`` of the time scores ``~a``.

So the first job is not accuracy, it is not collapsing.  Measured on the three
shipped checkpoints, the side head does collapse off-domain::

    P(the answer flips when the video is mirrored)   median |side logit|
    aihub-long   0.94 - 0.96                          7.2 - 7.5
    ccd-long     0.57 - 0.70                          1.3 - 1.6
    nexar-long   0.34 - 0.64                          0.7 - 1.7

A head that answers "which side did the victim come from" MUST flip its answer when
the image is mirrored.  On AI Hub imagery it flips 95% of the time, so it really is
reading geometry there.  On CCD / Nexar imagery the geometric part shrinks by 5x and
what survives is a mirror-INVARIANT offset -- a constant "say LEFT" bias -- which is
what produced LEFT 9 / RIGHT 3 on the twelve real folders.

THE FIX, AND WHY IT IS FREE
---------------------------
Write the head's output as ``s(x) = g(x) + b(x)`` where ``g`` is antisymmetric under a
mirror and ``b`` is invariant.  Then ``s(mirror x) = -g(x) + b(x)`` and

    s_sym(x) = (s(x) - s(mirror x)) / 2 = g(x)

removes the bias exactly, whatever its size, without retraining.  The mirror does not
have to be the true one: ``mirror_features`` swaps the CNN's left and right bands and
applies the published flow / frame-difference flip permutations, which is a pure
permutation of the 2382-d feature vector -- no second backbone pass, no second decode.
Measured against the exact mirror (``cnn_flip``, the mirrored image through the same
backbone) on 2201 stitched long clips: ``corr(s_sym_exact, s_sym_swap) = +0.99`` on
every corpus, the same LEFT/RIGHT answer 95-97% of the time on CCD / Nexar and 99% on
AI Hub, and AI Hub macro-F1 within 0.002 either way (0.9606/0.9596, 0.9656/0.9676,
0.9647/0.9626 for the three checkpoints).  The whole correction costs one extra forward
pass of a 4.45M-parameter net per ensemble member -- under 0.05 s against a folder that
takes 6.3 s end to end.

``evasion_space`` is not a left/right question, so symmetrising cannot help it; what
it has instead is a marginal that drifts off-domain.  The head emits ``1`` for 71.2%
of AI Hub clips against a 71.3% label base rate -- perfect in-domain calibration --
and for 30.7% / 44.7% of Nexar / CCD clips.  A single additive logit bias moves the
off-domain marginal back without disturbing the in-domain answer, because the AI Hub
logit distribution is five times wider than the shift: at ``EVASION_BIAS = +0.9`` the
real-corpus marginal goes 0.40 -> 0.54 while AI Hub out-of-fold macro-F1 moves
0.7497 -> 0.7492 (fold 0) and 0.7650 -> 0.7718 (fold 1).

A SECOND, INDEPENDENT VOTE
-------------------------
Symmetrising fixes the marginal but does not make the head see more: on 134 real clips
hand-labelled for this work (62 CCD, 72 of the real 1250-frame folders in
``build/bench/s2_nexar``) it is right 59.0% of the time against the shipped head's
59.7%.  ``geometric_side_score`` adds a cue that does not depend on the learned
representation at all -- which columns of the frame carry frame-difference energy in
the second before contact that they did not carry two seconds earlier, normalised by
the clip's own reference window -- and is right 64.9%.  The two are wrong on different
clips, so the shipped answer is their sum, which is right 64.9% pooled (63.9% on the
real folders alone, where the shipped head is 55.6%).  None of these differences is
significant on 134 clips (McNemar exact p = 0.31 against the shipped head); what IS
established is the marginal, which is 68% LEFT for the shipped head on those folders,
53% for this one, against a hand-labelled truth of 42% LEFT.

RULE 4
------
Every function here reads one clip and nothing else.  ``mirror_features`` is a
permutation of that clip's own features; ``EVASION_BIAS`` is a constant fitted once,
offline, on the training corpora -- a model parameter, like any weight -- and is never
re-derived from the evaluation set.  There is no ranking, no median, no quantile taken
across files anywhere in this module.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIDE_LABELS = ("LEFT", "RIGHT")

# Additive bias on ``logit(evasion=1) - logit(evasion=0)``.  See the module docstring:
# chosen so the out-of-domain marginal sits near the middle rather than at 0.40, which
# is the minimax choice while the private base rate is unknown.
EVASION_BIAS = 0.9

# The same constant, re-fitted for the FINE-TUNED backbone (``feature_meta`` says ft).
# It does NOT transfer, because it corrects a marginal and the fine-tuned features move
# that marginal the other way.  P(evasion = 1), symmetrised exactly as ``override``
# computes it, on the out-of-fold eval clips with the shipped trios:
#
#   WHOLE cached clips (one continuous recording -- the shape of a private folder)
#                  bias  -2.0   -1.5*  -1.0    0.0   +0.9
#     frozen  nexar 210         0.12   0.16   0.25   0.41      *interpolated
#     frozen  ccd   117         0.28   0.43   0.67   0.87
#     frozen  weighted                                0.574   <- what scored 0.52246
#     ft      nexar 210  0.47   0.55   0.64   0.75   0.82
#     ft      ccd   117  0.48   0.60   0.70   0.82   0.94
#     ft      weighted   0.474  0.57   0.661  0.775  0.863
#   40 s STITCHED views (nexar-long + ccd-long), ft trio: 0.474 / 0.58 / 0.68 / 0.83 / 0.89
#
# The frozen model needed +0.9 to lift 0.40 to 0.50 (stitched) / 0.57 (whole clip); the
# fine-tuned one already sits at 0.78-0.83 unbiased, so the shipped +0.9 drives it to
# 0.86-0.89 and makes the collapse WORSE.  -1.5 puts the fine-tuned real-corpus marginal
# back on the 0.574 that the leaderboard-verified frozen configuration actually ran at --
# the only calibration anchor there is.  It costs about 0.03 of AI Hub macro-F1
# (0.76 vs 0.789), the same kind of in-domain price +0.9 already paid for the frozen model
# (0.764 vs 0.783 at bias 0).
#
# SUPERSEDED 2026-09-03: the note below was answered by scripts/lever4_folders.py -- the
# 63 realistic ccdneg folders (built after this was written) show no collapse (median
# +2.35, p10 -0.07, p90 +4.25; -1.5 gives P(1)=0.619).  The ten folders it describes are a
# tiled video with a wrap-around cut the collision head fires on, so they were never a
# usable calibration set.  Kept for the record; do not act on it.
# NOT SETTLED: on the ten real 1250-frame folders in build/bench the fine-tuned evasion
# logits are four times smaller (median +0.43 against +1.82 on whole cached clips) and
# -1.5 sends 9 of 10 to class 0.  Those folders are a tiled video with a wrap-around cut
# at frame ~1210 that the fine-tuned collision head fires on, which poisons the
# event-anchored clip pooling -- so they are not a usable calibration set.  Before this
# constant ships, re-fit it on real continuous 1250-frame folders.
EVASION_BIAS_FT = -1.5


def evasion_bias(wants_ft: bool = False) -> float:
    """The additive evasion bias for the feature family the loaded net was trained on."""
    return EVASION_BIAS_FT if wants_ft else EVASION_BIAS

# Geometric vote: window (seconds relative to the predicted contact) in which the
# victim's arrival shows up as new frame-difference energy, and the reference window of
# the SAME clip it is compared against.  The windows are in the time base
# ``common.s2_assumed_fps`` gives the folder, like every other Stage-2 time.
GEO_PRE = (-1.1, -0.05)
GEO_REF = (-2.9, -1.4)
GEO_GRID = (24, 40)          # rows, cols of the block grid
# Weight of the geometric score when it is added to the symmetrised logit.  The two live
# on different scales -- the symmetrised logit has median |.| = 0.8-1.6 off-domain and 7.3
# on AI Hub, the centroid median |.| = 0.09 -- so the factor is large, and its value is what
# decides who wins where.  At 10 the geometric vote is worth about 0.9 logits: decisive
# off-domain, invisible in-domain.  Measured (macro-F1, out-of-fold where applicable):
#
#            weight     0 (head only)      5        10        20     geometry only
#   nexar-bench (72)        0.5751     0.6099    0.6364    0.6364      0.6108
#   ccd-long    (62)        0.5958     0.6591    0.6761    0.6761      0.6774
#   aihub fold0 (88)        0.9083     0.9083    0.9079    0.8961      0.6138
#   aihub fold1 (88)        0.8974     0.8974    0.9087    0.8737      0.6138
#
# 20 buys nothing off-domain over 10 and costs 0.01-0.02 in-domain, so 10 it is.
#
# LOWERED TO 5 (2026-08-29, LEVER 4).  ``geo`` is anchored on the PREDICTED contact frame
# and it is far more fragile to that anchor than the table above admits.  Measured on 63
# hand-labelled CCD clips -- CCD is the organisers' own Stage-2 source family: the five
# public Stage-2 example videos are byte-identical to data/ccd/00000{1..5}.mp4 (md5) -- with
# the head's out-of-fold symmetrised logit, macro-F1 as the anchor degrades:
#
#     anchor                 true   model's own   +0.6 s rms   +1.2 s rms
#     head alone            0.8083     0.8083       0.8083       0.8083
#     g + 10*geo (SHIPPED)  0.8056     0.8238       0.8083       0.7444
#     geo alone             0.6111     0.6636       0.6825       0.5074
#     g + 5*geo + 10*flow   0.8393     0.8238       0.8247       0.7918
#
# i.e. the shipped weight is fine when the collision head is right and costs 0.06 when it is
# not, and on a 1250-frame private folder the collision head is at 0.62-0.71 Acc@+-0.3 s,
# worse than the 0.744 it manages on these 5 s clips.  Halving the weight and adding a
# second, differently-fragile geometric vote (``flow_side_score``) is at least as good as
# the shipped rule at EVERY anchor quality measured and 0.05 better at the bad end.
#
# READ THAT TABLE AS SELECTION-OPTIMISTIC, not as an out-of-sample measurement (verified
# 2026-08-28):
#   * the weight AND the window/normalisation pair below were both chosen on the same 134
#     hand labels the rows report.  The shipped (GEO_PRE, GEO_REF, ratio) pair is rank 1 of
#     the 96 configurations tried, pooled accuracy 0.6493; the MEDIAN configuration scores
#     0.6194 and a leave-one-corpus-out choice 0.5896.  Take ~0.62 as the honest geometry-only
#     number and ~0.61-0.63 (vs 0.5967 for the shipped head) for the sum.
#   * the nexar-bench row was computed with the EXACT mirror (cnn_flip through the backbone).
#     What ships is the band swap, which scored 0.6084 there, not 0.6364.
#   * the cue is anchored on the PREDICTED collision frame, and those 72 folders were inside
#     best.pt's training set, so the anchor was perfect (Acc@+-0.3 s = 1.000).  Out of fold
#     the collision head is right 46% of the time on nexar-long; re-scoring the same clips
#     with a 1 s r.m.s. anchor error takes geometry-only from 0.649 to 0.577 and the sum
#     from 0.657 to 0.608.
GEO_WEIGHT = 5.0

# Weight of the flow vote.  ``flow_side_score`` returns a lateral offset in image widths --
# the same unit ``geometric_side_score`` returns -- but its magnitude is about half, so 10
# makes the two votes comparable in influence.  On the 63 CCD hand labels the pair
# (5, 10) is the only combination of the ones tried that never loses to the shipped
# (10, 0) at any of the four anchor qualities above.
FLOW_WEIGHT = 10.0


# --------------------------------------------------------------------------------------
# The free mirror
# --------------------------------------------------------------------------------------
# Two different band ORDERINGS live in this pipeline, and they are not interchangeable.
#
#   "band"  stage2_features.BandExtractor: ``stack(bands, 1).reshape(B, N_BANDS*C)``
#           -> [b0c0 .. b0c767, b1c0 .. b1c767, b2c0 .. b2c767]
#   "chan"  stage2_backbone.ImpactNet.bands: ``adaptive_avg_pool2d(f, (1, N_BANDS)).flatten(1)``
#           -> [c0b0, c0b1, c0b2, c1b0, c1b1, c1b2, ...]
#
# Both are 3*C wide, both divide by three, and mirroring one with the other's rule
# silently scrambles channels instead of swapping sides.  Measured on 40 frames of a real
# 1250-frame folder against the EXACT mirror (the flipped image through the same
# backbone), correlation of the permuted vector with the true mirrored vector:
#
#                          fine-tuned block      frozen block
#     band-major swap           +0.807              +0.980
#     channel-major swap        +0.988              +0.307
#     identity (no swap)        +0.976                --
#
# i.e. on the fine-tuned block the band-major swap this module shipped is WORSE than not
# mirroring at all, so ``s_sym = (s(x) - s(mirror x))/2`` was subtracting noise rather
# than the mirror-invariant bias.  ``predict_folder`` now passes the exact mirror instead;
# this permutation stays as the fallback for the frozen path and is order-aware.
BAND_ORDER_BAND = "band"
BAND_ORDER_CHAN = "chan"


def block_spec(in_dim: int, device: str = "cpu", wants_ft: bool = False):
    """``[(width, order), ...]`` of the visual sub-blocks ``stage2_model._cnn_block`` emits.

    Mirrors that function's three-way decision and additionally records which band
    ORDERING each sub-block uses, which is the part ``stage2_model._cnn_layout`` cannot
    express.  Returns ``None`` when the layout cannot be resolved, which makes the caller
    fall back to the historic single-block band-major reading.
    """
    try:
        import stage2_features as S2F
        import stage2_model as S2M

        rest = S2F.FLOW_DIM + S2F.FD_DIM
        cnn_dim = int(in_dim) - rest
        if cnn_dim <= 0:
            return None
        ext, _ = S2M._ft_extractor(device) if wants_ft else (None, None)
        if ext is not None:
            ft = int(getattr(ext, "feat_dim", 0))
            pool = getattr(getattr(ext, "net", None), "pool", "mean")
            parts = [(ft // 2, BAND_ORDER_CHAN), (ft // 2, BAND_ORDER_CHAN)] \
                if (pool == "meanmax" and ft % 2 == 0) else [(ft, BAND_ORDER_CHAN)]
            if cnn_dim == ft:
                return parts
            if cnn_dim == ft + 2304:
                return parts + [(2304, BAND_ORDER_BAND)]
            return None
        return [(cnn_dim, BAND_ORDER_BAND)]
    except Exception:                                   # noqa: BLE001
        return None


def mirror_features(X: np.ndarray, cnn_blocks=None) -> np.ndarray:
    """Approximate horizontal mirror of an assembled ``[T, FEATURE_DIM]`` matrix.

    The CNN block is three horizontal bands (left / centre / right), so the mirror swaps
    bands 0 and 2; the flow block has a published flip plan
    (``stage3_features.flip_features``) and the frame-difference block a permutation
    (``stage2_features.FD_FLIP_PERM``).  Only the CNN part is approximate -- the exact
    mirror pushes the mirrored image through the backbone, which is what
    ``stage2_decode.predict_folder`` now hands in directly.

    ``cnn_blocks`` is the width of each CONCATENATED visual sub-block, in order, and each
    entry may be a bare width (band-major, the historic reading) or a ``(width, order)``
    pair with ``order`` in ``{"band", "chan"}``.  It matters because
    ``stage2_model._cnn_block`` has three layouts and only one of them is a single
    band-major 3-band block: reshaping a stacked 4608-wide vector into three 1536-wide
    "bands" divides exactly and mirrors NOTHING correctly, and the fine-tuned block is
    channel-major so even a correctly SIZED band swap scrambles it.  A layout that does
    not add up raises here, which makes ``override`` return None and leaves the caller's
    own answer in place.  ``None`` keeps the historical single-block band-major reading.
    """
    import stage2_features as S2F
    import s2flow as S3F

    X = np.asarray(X, dtype=np.float32)
    # The layout contract is [ visual sub-blocks | flow(FLOW_DIM) | fd(FD_DIM) ], and the
    # visual width is read off the ARRAY rather than from S2F.CNN_DIM so that a different
    # backbone -- a fine-tuned one with a wider or narrower band, say -- still mirrors
    # correctly as long as each sub-block keeps the three horizontal bands.
    flow_dim, fd_dim = S2F.FLOW_DIM, S2F.FD_DIM
    cnn_dim = int(X.shape[-1]) - flow_dim - fd_dim
    if cnn_dim <= 0:
        raise ValueError(f"unexpected Stage-2 feature layout: width {X.shape[-1]}")
    spec = [(int(cnn_dim), BAND_ORDER_BAND)] if cnn_blocks is None else \
        [((int(b), BAND_ORDER_BAND) if np.isscalar(b) else (int(b[0]), str(b[1])))
         for b in cnn_blocks]
    if sum(b for b, _ in spec) != cnn_dim or any(b <= 0 or b % S2F.N_BANDS for b, _ in spec):
        raise ValueError(f"visual sub-blocks {spec} do not tile {cnn_dim} channels "
                         f"in {S2F.N_BANDS} bands each")
    out = X.copy()
    off = 0
    for b, order in spec:
        sl = X[..., off:off + b]
        if order == BAND_ORDER_CHAN:
            sub = sl.reshape(X.shape[:-1] + (b // S2F.N_BANDS, S2F.N_BANDS))
            out[..., off:off + b] = sub[..., ::-1].reshape(X.shape[:-1] + (b,))
        else:
            sub = sl.reshape(X.shape[:-1] + (S2F.N_BANDS, b // S2F.N_BANDS))
            out[..., off:off + b] = sub[..., ::-1, :].reshape(X.shape[:-1] + (b,))
        off += b
    out[..., cnn_dim:cnn_dim + flow_dim] = S3F.flip_features(X[..., cnn_dim:cnn_dim + flow_dim])
    out[..., cnn_dim + flow_dim:] = X[..., cnn_dim + flow_dim:][..., S2F.FD_FLIP_PERM]
    return out


def symmetric_logits(members, x, mask=None, cnn_blocks=None, x_mirror=None
                     ) -> tuple[float, float]:
    """``(side_logit_symmetrised, evasion_logit_averaged)`` for one clip.

    ``members`` is the list of nets ``predict_folder`` already built, ``x`` the
    ``[1, T, D]`` tensor it already fed them.  Returns the two clip-level logit
    differences (class 1 minus class 0), the side one with its mirror-invariant part
    removed and the evasion one averaged over both orientations to cut variance.

    ``x_mirror`` is the EXACT mirror when the caller has it -- the flipped frames pushed
    through the same backbone, which ``stage2_decode.extract_blocks`` computes anyway for
    flip TTA.  When it is absent the permutation above stands in.
    """
    import torch

    device = x.device
    if x_mirror is None:
        xm = torch.from_numpy(mirror_features(x.detach().cpu().numpy(), cnn_blocks)).to(device)
    else:
        xm = x_mirror if torch.is_tensor(x_mirror) else \
            torch.from_numpy(np.asarray(x_mirror, dtype=np.float32)).to(device)
    if mask is None:
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=device)
    s_sum = v_sum = 0.0
    with torch.no_grad():
        for m in members:
            a = m(x, mask)
            b = m(xm, mask)
            sa = float(a["side"][0, 1] - a["side"][0, 0])
            sb = float(b["side"][0, 1] - b["side"][0, 0])
            va = float(a["evasion"][0, 1] - a["evasion"][0, 0])
            vb = float(b["evasion"][0, 1] - b["evasion"][0, 0])
            s_sum += 0.5 * (sa - sb)
            v_sum += 0.5 * (va + vb)
    n = max(1, len(members))
    return s_sum / n, v_sum / n


# --------------------------------------------------------------------------------------
# Second vote: where did new motion appear, on this clip's own baseline
# --------------------------------------------------------------------------------------
def geometric_side_score(frame_paths, frame_numbers, fps: float, collision_frame: float,
                         step: float = 0.1) -> float | None:
    """``> 0`` means the colliding vehicle arrived from the RIGHT; ``None`` if unusable.

    Reads ~30 frames around the predicted contact at reduced resolution, builds a block
    map of ``|I_t - I_{t-1}|``, and asks which columns carry frame-difference energy in
    the second before impact that they did not carry two seconds earlier.  Normalising
    each window by its own total is what makes this a per-file statement: the answer is
    a comparison of one clip against itself, and ego speed, exposure and camera cancel.

    Measured on 134 hand-labelled real clips (62 CCD, 72 real 1250-frame Nexar folders)
    this is right 64.9% of the time against 59.7% for the shipped head, and 61.9% is the
    MEDIAN over all 96 window/normalisation configurations tried -- i.e. the signal is a
    property of the cue, not of the tuning.  It is balanced by construction (it is a
    centroid of a non-negative map, so nothing pins its sign), which is what stops the
    macro-F1 collapse even on the clips where it is wrong.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
        n = len(frame_paths)
        if n < 8 or fps <= 0:
            return None
        gy, gx = GEO_GRID
        blk = 8
        offs = np.arange(GEO_REF[0], GEO_PRE[1] + 1e-6, step)
        numbers = np.asarray(frame_numbers, dtype=np.float64)
        want_no = float(collision_frame) + offs * fps
        idx = np.unique(np.clip(np.searchsorted(numbers, want_no), 0, n - 1))
        if len(idx) < 8:
            return None
        small, t_rel = [], []
        for j in idx:
            # half-resolution decode: this is a 40x24 block map in the end, so decoding
            # the jpeg at full size would be ~4x the cost for no change in the answer.
            img = cv2.imread(str(frame_paths[int(j)]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if img is None:
                continue
            small.append(cv2.resize(img, (gx * blk, gy * blk), interpolation=cv2.INTER_AREA))
            t_rel.append((numbers[int(j)] - float(collision_frame)) / fps)
        if len(small) < 8:
            return None
        return geometric_side_score_from_grays(small, t_rel)
    except Exception:                                   # noqa: BLE001
        return None


def geometric_side_score_from_grays(small, t_rel) -> float | None:
    """The block-map half of ``geometric_side_score``, on frames already in memory.

    ``small`` are grayscale frames at ``GEO_GRID * 8`` pixels, in order; ``t_rel`` their
    times relative to contact.  Split out so the evaluation harness can feed the same
    arrays to this and to ``flow_side_score_from_grays``, and can mirror them, without
    writing a temporary JPEG per frame.
    """
    try:
        gy, gx = GEO_GRID
        blk = 8
        if len(small) < 8:
            return None
        S = np.stack(small)
        t = np.asarray(t_rel)[1:]
        d = np.abs(S[1:].astype(np.int16) - S[:-1].astype(np.int16)).astype(np.float32)
        d = d.reshape(d.shape[0], gy, blk, gx, blk).mean(axis=(2, 4))
        pre = (t >= GEO_PRE[0]) & (t <= GEO_PRE[1])
        ref = (t >= GEO_REF[0]) & (t <= GEO_REF[1])
        if pre.sum() < 2 or ref.sum() < 2:
            return None
        P = d[pre].mean(0)
        R = d[ref].mean(0)
        # Ratio, not difference: dividing by the clip's own reference map is what removes
        # the ego-motion pattern (the road streaming past, the parked cars at the edges)
        # and leaves only the columns that got busier because something arrived in them.
        D = np.clip(P / (R + 0.5 * float(R.mean()) + 1e-6) - 1.0, 0.0, None)
        if D.sum() <= 1e-9:
            return None
        xs = (np.arange(gx) + 0.5) / gx - 0.5
        return float((D.sum(0) * xs).sum() / D.sum())
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
# Third vote: the moving object's lateral offset from the lane centreline (LEVER 4)
# --------------------------------------------------------------------------------------
# ``geometric_side_score`` above compares one clip's frame-difference map against its own
# earlier map.  That removes the ego pattern only to the extent that the ego pattern did
# not change, and it measures offset from the IMAGE centre, which is not the lane
# centreline on a camera that is not centred or an ego car that is not straight.
#
# ``flow_side_score`` answers the same question the AI Hub label answers -- "sign of the
# victim's lateral offset from the lane centreline, in the window before entry" -- with
# optical flow instead of boxes:
#
#   1. DIS flow between consecutive frames of the pre-collision window;
#   2. the focus of expansion of that flow (``stage2_entry.estimate_foe``) is the lane
#      centreline AND the origin of the ego-motion field, both at once;
#   3. every static point's flow is COLLINEAR with (p - foe), so the perpendicular
#      component of the flow about the FOE is zero for the whole static world and large
#      exactly on the independently-moving vehicles.  A global rotation (the ego turning)
#      adds a constant to that perpendicular component, so subtracting its weighted median
#      removes the turn;
#   4. the answer is the residual-weighted mean of ``(u - u_foe)``, the same normalised
#      image-width offset from the lane centre that ``stage2_aihub.derive`` takes the
#      median of over its five pre-entry anchors.
#
# Antisymmetric by construction: mirroring the frames maps u -> 1-u and the FOE with it,
# flips the sign of every perpendicular residual but not its magnitude, so the weighted
# mean offset changes sign and nothing else.  Measured, not assumed -- see
# ``scripts/lever4_geom.py``, which runs the whole function on mirrored frames.
FLOW_PRE = (-2.2, -0.25)     # seconds relative to contact: the victim is still off-axis
FLOW_MAX_PAIRS = 22
FLOW_SIZE = (320, 180)


def _flow_frames(frame_paths, frame_numbers, fps: float, collision_frame: float,
                 window, step: float, size):
    """Grayscale frames of one clip inside ``window`` seconds of contact, plus times."""
    import cv2

    n = len(frame_paths)
    if n < 4 or fps <= 0:
        return None, None
    offs = np.arange(window[0], window[1] + 1e-6, step)
    numbers = np.asarray(frame_numbers, dtype=np.float64)
    want = float(collision_frame) + offs * fps
    idx = np.unique(np.clip(np.searchsorted(numbers, want), 0, n - 1))
    if len(idx) < 4:
        return None, None
    grays, t_rel = [], []
    for j in idx:
        img = cv2.imread(str(frame_paths[int(j)]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            continue
        grays.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        t_rel.append((numbers[int(j)] - float(collision_frame)) / fps)
    if len(grays) < 4:
        return None, None
    return grays, np.asarray(t_rel, float)


def flow_side_score_from_grays(grays, t_rel, max_pairs: int = FLOW_MAX_PAIRS
                               ) -> tuple[float, float] | None:
    """``(signed offset in image widths, residual mass)``; ``> 0`` means RIGHT.

    Separated from the decoding so the same code can be run on a video, on a folder of
    JPEGs and on the horizontally mirrored version of either.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
        import sys as _s
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in _s.path:
            _s.path.insert(0, _here)
        import stage2_entry as S2E

        n = len(grays)
        if n < 4:
            return None
        H, W = grays[0].shape[:2]
        geom = S2E.estimate_foe(grays)
        fx, fy = geom.u_vp * W, geom.v_vp * H
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        step = max(1, (n - 1) // max_pairs)
        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - fx
        dy = ys - fy
        rad = np.hypot(dx, dy) + 1e-6
        # only the ground plane: above the horizon there is sky and canopy, and the
        # victim is by definition on the road
        below = ys > (fy + 0.02 * H)
        num = 0.0
        den = 0.0
        mass = 0.0
        for i in range(step, n, step):
            fl = dis.calc(grays[i - step], grays[i], None)
            u = fl[..., 0].astype(np.float32)
            v = fl[..., 1].astype(np.float32)
            # component of the flow perpendicular to the ray from the FOE: zero for
            # every static point, whatever the ego speed
            perp = (u * dy - v * dx) / rad
            m = below & (np.hypot(u, v) > 0.35)
            if int(m.sum()) < 64:
                continue
            # a turning ego adds a constant perpendicular field -- take it out
            perp = perp - float(np.median(perp[m]))
            w = np.where(m, np.abs(perp), 0.0).astype(np.float64)
            w = np.clip(w - 0.30, 0.0, None)          # DIS noise floor
            s = float(w.sum())
            if s <= 1e-6:
                continue
            num += float((w * (xs - fx)).sum()) / W
            den += s
            mass += s / (H * W)
        if den <= 1e-6:
            return None
        return float(num / den), float(mass / max(1, n))
    except Exception:                                   # noqa: BLE001
        return None


def flow_side_score(frame_paths, frame_numbers, fps: float, collision_frame: float,
                    step: float = 0.1, window=FLOW_PRE) -> float | None:
    """``> 0`` means the colliding vehicle arrived from the RIGHT; ``None`` if unusable."""
    try:
        grays, t_rel = _flow_frames(frame_paths, frame_numbers, fps, collision_frame,
                                    window, step, FLOW_SIZE)
        if grays is None:
            return None
        r = flow_side_score_from_grays(grays, t_rel)
        return None if r is None else r[0]
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
# evasion_space: measure the free space instead of asking the head (LEVER 4, day 3)
# --------------------------------------------------------------------------------------
# WHY THE HEAD IS NOT THE RIGHT SOURCE FOR THIS FIELD
# ---------------------------------------------------
# Every evasion number this project had before today was measured on CACHED features -- a
# 5 s clip's whole feature matrix, or a 40 s stitching of cached chunks.  The private set
# is neither: it is a folder of 1250 JPEGs that ``predict_folder`` re-extracts features
# from and pools over.  ``scripts/lever4_folders.py`` builds folders of that exact shape
# out of clips that have labels -- a CCD accident dropped at a random position inside
# Nexar no-collision driving, 1250 frames at the 10 fps time base ``assumed_fps`` assigns
# -- and runs the shipped entry point on them.  Two things came out of 63 of them:
#
#   * the pooled evasion logit is median +2.35 with p10 -0.07 and p90 +4.25, so it does
#     vary per file and the additive bias does control the marginal: -1.5 gives
#     P(evasion = 1) = 0.619, 0.0 gives 0.873, +0.9 gives 0.968.  (Build the same folders
#     with 24 OTHER CRASHES as filler instead of ordinary driving and the logit collapses
#     to median +2.97 / p10 +2.25 / p90 +3.63, a constant no bias can split -- worth
#     knowing, because it says the pooled logit is reading the FILLER as much as the
#     accident.)
#   * what it does NOT have is information.  Macro-F1 over two classes is capped at 0.5
#     for any predictor that carries none, whatever its marginal
#     (``pi*q/(pi+q) + (1-pi)*(1-q)/(2-pi-q) <= 0.5``, equality at q = pi), and the
#     leaderboard arithmetic puts this field at 0.47-0.50 -- exactly that ceiling.  The
#     head scores 0.79 against the AI Hub derived label it was trained on and nothing
#     above chance off it.  No re-tuning of ``EVASION_BIAS_FT`` can move a number that is
#     already at the no-information ceiling; only per-file information can.
#
# WHAT REPLACES IT
# ----------------
# The label's own definition is geometric -- "was there room to avoid" -- so measure it:
# count road users in the two lanes either side of the ego lane in the second before
# contact, and call it free when the EMPTIER of the two is clear.  Measured on 499
# out-of-fold AI Hub clips against the derived label (base rate 0.717,
# ``scripts/lever4_evasion.py --min-conf 0``; "uninf." is the macro-F1 of a predictor with
# the same marginal and NO information, ``pi*q/(pi+q) + (1-pi)*(1-q)/(2-pi-q)``, so "gain"
# is the part that is real):
#
#     predictor                                    macroF1   P(1)   uninf.   gain
#     constant 1                                    0.4177  1.000   0.4177  +0.000
#   SSDLite320 counts (what this ships)
#     min(n_left, n_right) == 0                     0.7189  0.739   0.4997  +0.219
#     min == 0 and n_all <= 2      <-- SHIPPED      0.7329  0.691   0.4996  +0.233
#     min == 0 and n_all <= 1                       0.6567  0.561   0.4868  +0.170
#     min(n_left, n_right) <= 1                     0.5284  0.958   0.4468  +0.082
#     n_all <= 1 (an uncrowded scene, no lanes)     0.6428  0.611   0.4937  +0.149
#     away lane == 0 (the label's own rule)         0.7816  0.653   0.4976  +0.284
#   Faster R-CNN R50-FPN counts, same clips, same rules
#     min(n_left, n_right) == 0                     0.5806  0.361   0.4360  +0.145
#     min(n_left, n_right) <= 1                     0.6579  0.639   0.4965  +0.161
#     away lane <= 1                                0.6982  0.535   0.4822  +0.216
#   the head, for reference, on the 5 s CACHED clip (unavailable on a folder)
#     head, bias 0                                  0.7930  0.768   0.4984  +0.295
#
# THE COARSE DETECTOR IS THE BETTER ONE, and not by a little: SSDLite320 beats Faster
# R-CNN by 0.14 macro-F1 on the shipped rule.  It sees a median of 1 road user per frame
# against Faster R-CNN's 8, because 320x320 input drops everything small -- and everything
# small is far away, and a car 80 m down the neighbouring lane has nothing to do with
# whether there was room to swerve NOW.  The crude detector is measuring the right
# quantity.  It is also 14 MB against 168 MB and ~20x faster, so it is what travels in the
# bundle.
#
# WHY THE SIDE-FREE RULE AND NOT THE LABEL'S OWN AWAY-LANE RULE
# ------------------------------------------------------------
# The away-lane rule is better only when ``entry_side`` is right, and it is fed OUR side
# prediction.  Re-scored with the side corrupted at the rate our own head errs:
#
#     side error   0.00     0.10     0.20     0.30
#     away == 0   0.7816   0.7399   0.7057   0.6668
#
# against 0.7189 for the side-free rule at every one of them.  Our side is right about
# 72-81% of the time, so the two are a tie at best and the side-free rule cannot be
# poisoned by a side error -- and a coupled failure (side wrong -> evasion wrong on the
# same file) is exactly the correlation a two-field score punishes twice.
#
# WHAT THIS IS AND IS NOT WORTH.  The status quo is NOT a collapsed constant: on the 63
# realistic folders the head at bias -1.5 answers 1 for 0.619 of them, and an
# uninformative predictor at that marginal scores 0.487-0.500 for any base rate in
# [0.5, 0.8] -- which is exactly the 0.47-0.50 the leaderboard attributes to the field.
# So this change is worth NOTHING under the null and up to +0.23 macro-F1 if the measured
# in-domain gain transfers; it is a bet on the transfer, not a free repair.  What makes
# the bet a reasonable one is that the cue is a COCO detector and a fixed lane triangle,
# neither of which was fitted to this task, and that its marginal barely moves across
# domains: P(evasion = 1) is 0.691 on AI Hub clips, 0.810 on CCD clips and 0.778 on the
# 63 real 1250-frame folders, against a head whose marginal moves 0.62 -> 0.97 with a
# 2.4-logit change of bias.  A marginal that stable is the signature of a measurement.
#
# RULE 4: every count is read from one clip's own frames.  Nothing is ranked, no quantile
# is taken across files, and ``EVASION_MAX_NEIGHBOUR`` is a constant fitted once, offline,
# on AI Hub -- a model parameter like any weight.
EVASION_MAX_NEIGHBOUR = 0
# ...AND the scene is not crowded.  The second clause is what keeps the marginal off the
# ceiling: on the 499 AI Hub clips ``min == 0`` alone answers 1 for 0.739 of files and
# ``min == 0 and n_all <= 2`` for 0.691, at a HIGHER macro-F1 (0.7329 vs 0.7189, gain
# +0.233 vs +0.219).  The marginal matters because the private base rate is unknown: the
# macro-F1 an uninformative predictor with marginal q scores is 0.481-0.500 across
# pi in [0.5, 0.8] at q = 0.691 and 0.470-0.499 at q = 0.739.  Ten rules were tried on
# those 499 clips and this was the best of them, so read the 0.014 as selection, not as
# an out-of-sample gain; what is NOT selection is the marginal, which moves the floor.
EVASION_MAX_SCENE = 2
EVASION_WINDOW = (-1.0, 0.0)       # seconds before contact the lanes must be free in
EVASION_FOE_WINDOW = (-2.5, 0.0)   # the window the vanishing point is estimated over
EVASION_N_FRAMES = 5
EVASION_SCORE = 0.30               # detector confidence floor the table above was scored at

# COCO ids: 1 person, 2 bicycle, 3 car, 4 motorcycle, 6 bus, 8 truck.  torchvision's COCO
# detectors emit the 91-class ids, in which these are 1/2/3/4/6/8.
COCO_ROAD_USER = frozenset({1, 2, 3, 4, 6, 8})

_DETECTOR: dict = {}


def _detector_weight_path(fn: str) -> str | None:
    """Where the packaged COCO detector weights are, or ``None``.

    Mirrors ``stage3_features.default_weight_path``: the training machine may download
    once, the evaluation server never can, so the file has to travel inside the bundle.
    """
    env = os.environ.get("STAGE2_DET_DIR")
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [env] if env else []
    for base in (here, os.path.dirname(here)):
        cands += [os.path.join(base, "model", "stage2"),
                  os.path.join(base, "models", "stage2"),
                  os.path.join(base, "model", "stage3"),
                  os.path.join(base, "models", "pretrained"),
                  os.path.join(base, "models"), base]
    cands.append(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    for c in cands:
        if not c:
            continue
        p = os.path.join(c, fn)
        if os.path.isfile(p):
            return p
    return None


def load_detector(device: str = "cpu", arch: str | None = None):
    """A COCO detector for the free-space count, or ``None`` if none is packaged.

    ``None`` is not an error: ``decide_evasion`` then keeps the head's answer, exactly as
    it did before this function existed.
    """
    arch = arch or os.environ.get("STAGE2_DET_ARCH", "ssdlite")
    key = (arch, str(device))
    if key in _DETECTOR:
        return _DETECTOR[key]
    model = None
    try:
        import torch
        import torchvision.models.detection as D

        files = {"fasterrcnn": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
                 "ssdlite": "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"}
        path = _detector_weight_path(files.get(arch, ""))
        if path is None:
            print(f"[stage2_side] no {arch} weights packaged -> evasion keeps the head",
                  file=sys.stderr, flush=True)
        else:
            if arch == "ssdlite":
                model = D.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None,
                                                        num_classes=91)
            else:
                model = D.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                                  num_classes=91, min_size=640, max_size=1138)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model = model.eval().to(device)
    except Exception as exc:                                # noqa: BLE001
        print(f"[stage2_side] detector unavailable ({type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)
        model = None
    _DETECTOR[key] = model
    return model


def _neighbour_counts(boxes, labels, geom, W: int, H: int, margin: float = 0.15):
    """``(n_left, n_right)`` road users in the two lanes flanking the ego lane.

    The ego lane is the triangle ``|u - u_vp| <= k*(v - v_vp)``; parallel lanes share the
    vanishing point, so the neighbour lane is the same triangle translated by
    ``2*k*(v - v_vp)`` -- the shift grows with ``v`` at the rate the lane widens, which is
    what makes this one comparison of a clip against its own perspective.
    """
    n_l = n_r = 0
    for b, lb in zip(boxes, labels):
        if int(lb) not in COCO_ROAD_USER:
            continue
        u = 0.5 * (float(b[0]) + float(b[2])) / W
        v = min(1.0, float(b[3]) / H)
        if v <= geom.v_vp:
            continue
        half = max(geom.half(v), 1e-6)
        if abs(u - (geom.u_vp - 2.0 * half)) <= half * (1.0 + margin):
            n_l += 1
        if abs(u - (geom.u_vp + 2.0 * half)) <= half * (1.0 + margin):
            n_r += 1
    return n_l, n_r


def free_space_counts(frame_paths, frame_numbers, fps: float, collision_frame: float,
                      device: str = "cpu", arch: str | None = None):
    """``(n_left, n_right, n_all)`` road users around contact, or ``None``.

    ``n_left`` / ``n_right`` are the MAXIMUM over the window -- a lane counts as blocked
    if any frame of the second before impact has a road user in it, which is the rule the
    AI Hub label was derived under -- and ``n_all`` is the MEDIAN over the window of the
    road users visible anywhere in the frame, i.e. how crowded this particular scene is.
    Split out from the decision so the rule can be re-scored offline without re-running
    the detector.
    """
    try:
        import cv2
        import torch

        cv2.setNumThreads(1)
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import stage2_entry as S2E

        model = load_detector(device, arch)
        if model is None or not frame_paths or not fps:
            return None
        n = len(frame_paths)
        numbers = np.asarray(frame_numbers, dtype=np.float64)

        def pick(window, count):
            offs = np.linspace(window[0], window[1], count)
            want = float(collision_frame) + offs * float(fps)
            return np.unique(np.clip(np.searchsorted(numbers, want), 0, n - 1))

        # the vanishing point, on the same 0.15 s grid stage2_entry was tuned for
        foe_idx = pick(EVASION_FOE_WINDOW, 18)
        grays = []
        for j in foe_idx:
            g = cv2.imread(str(frame_paths[int(j)]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if g is None:
                continue
            grays.append(cv2.resize(g, (480, 270), interpolation=cv2.INTER_AREA))
        if len(grays) < 4:
            return None
        geom = S2E.estimate_foe(grays)

        rgb = []
        for j in pick(EVASION_WINDOW, EVASION_N_FRAMES):
            img = cv2.imread(str(frame_paths[int(j)]), cv2.IMREAD_COLOR)
            if img is not None:
                rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(rgb) < 2:
            return None
        H, W = rgb[0].shape[:2]
        n_l = n_r = 0
        totals = []
        with torch.inference_mode():
            ten = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)
                   .to(device).float().div_(255.0) for f in rgb]
            for pr in model(ten):
                k = pr["scores"] >= EVASION_SCORE
                lab = pr["labels"][k].cpu().numpy()
                a, b = _neighbour_counts(pr["boxes"][k].cpu().numpy(), lab, geom, W, H)
                n_l, n_r = max(n_l, a), max(n_r, b)
                totals.append(sum(1 for x in lab if int(x) in COCO_ROAD_USER))
        return int(n_l), int(n_r), int(np.median(totals)) if totals else 0
    except Exception as exc:                                # noqa: BLE001
        print(f"[stage2_side] free_space_counts failed "
              f"({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return None


def free_space_evasion(frame_paths, frame_numbers, fps: float, collision_frame: float,
                       device: str = "cpu", arch: str | None = None) -> int | None:
    """``1`` when there was room to avoid, ``0`` when there was not, ``None`` if unusable.

    ``None`` -- no packaged detector, unreadable frames, a degenerate window -- leaves the
    caller's own answer in place, which is the head-plus-bias answer that shipped before.
    """
    r = free_space_counts(frame_paths, frame_numbers, fps, collision_frame, device, arch)
    if r is None:
        return None
    n_l, n_r, n_all = r
    return int(min(n_l, n_r) <= EVASION_MAX_NEIGHBOUR and n_all <= EVASION_MAX_SCENE)


# --------------------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------------------
def decide_side(side_logit: float | None, geo_score: float | None = None,
                flow_score: float | None = None,
                geo_weight: float = GEO_WEIGHT,
                flow_weight: float = FLOW_WEIGHT) -> str:
    """``side_logit`` is the SYMMETRISED logit (class RIGHT minus class LEFT).

    Three votes, added rather than gated, because they fail on different clips and for
    different reasons: the head needs no anchor at all and is flat at macro-F1 0.808 on
    the 63 hand-labelled CCD clips whatever the collision head does; ``geo`` and ``flow``
    both read the frames around the PREDICTED contact and both decay as that prediction
    decays, but not at the same rate (``geo`` compares two windows so a shift moves both,
    ``flow`` reads one window so a shift walks it past the impact).  Summing all three is
    what keeps the answer above the head-alone floor when the anchor is good without
    falling below it when the anchor is bad.

    A missing vote contributes nothing rather than a zero-with-weight, so a clip whose
    frames could not be read still gets the head's answer.
    """
    total = 0.0
    have = False
    if side_logit is not None and np.isfinite(side_logit):
        total += float(side_logit)
        have = True
    if geo_score is not None and np.isfinite(geo_score):
        total += float(geo_weight) * float(geo_score)
        have = True
    if flow_score is not None and np.isfinite(flow_score):
        total += float(flow_weight) * float(flow_score)
        have = True
    if not have:
        return SIDE_LABELS[0]
    return SIDE_LABELS[1] if total >= 0.0 else SIDE_LABELS[0]


def decide_evasion(ev_logit: float | None, bias: float = EVASION_BIAS,
                   free: int | None = None) -> int:
    """The measured free space when there is one, else the head with its additive bias.

    ``free`` wins outright rather than being blended in, because on folder-shaped input
    the head is a constant (see the block above): a blend would only re-add that constant
    to every file.  When no detector is packaged ``free`` is ``None`` and this is exactly
    the function that shipped before.
    """
    if free is not None:
        return int(free)
    if ev_logit is None or not np.isfinite(ev_logit):
        return 1
    return int((float(ev_logit) + float(bias)) >= 0.0)


# --------------------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------------------
# ``use_free_space`` gates the SAME SSDLite free-space-evasion measurement that
# ``stage2_decode.FREE_SPACE_EVASION`` gates, but this is a second, independent call path
# into it: stage2_model.predict_folder (the LEVER 4 hook above it, and the per-folder /
# whole-run rescue that stage2_decode and stage2_infer fall back to whenever the primary
# decoder throws or is absent) calls this function without ever passing
# ``use_free_space=False``, so a default of True here would fire regardless of
# STAGE2_FREE_SPACE. Defaults OFF as of 2026-08-30 for the identical reason
# FREE_SPACE_EVASION was turned off: the adversarial verify on lever 4 found the +0.233
# macro-F1 gain is measured on a tautological surface (SSDLite is scored against the very
# AI Hub box-annotation rule it approximates, not against organiser labels), that on that
# same surface the incumbent head it replaces already scores HIGHER (0.798 vs 0.733), and
# that the claimed "nothing to lose" floor was understated 3-5x once a day/night split of
# the same corpus showed the rule's marginal moving 0.79->0.92 on illumination alone. Do
# not flip this default without a validation surface that is not also the training
# label's own rule -- and if it is ever flipped, flip it here AND in every path that can
# call this function, not just the one that was flipped last time.
def override(members, x, mask=None, frame_paths=None, frame_numbers=None,
             fps: float | None = None, collision_frame: float | None = None,
             use_geometry: bool = True, cnn_blocks=None,
             wants_ft: bool = False, x_mirror=None,
             evasion_bias_override: float | None = None,
             use_free_space: bool = False, device: str | None = None) -> dict | None:
    """Everything ``predict_folder`` needs, or ``None`` if anything goes wrong.

    Never raises: a failure here must leave the caller's own answers in place, exactly
    like a failure in ``predict_folder`` leaves the heuristic in place.
    """
    try:
        if not members or x is None:
            return None
        # ``cnn_blocks`` from stage2_model._cnn_layout carries widths only, and widths
        # alone cannot say whether a 2304-wide visual block is band-major (frozen) or
        # channel-major (fine-tuned).  Resolve the layout here, where the ordering is
        # known, and fall back to what the caller passed if that fails.
        if x_mirror is None:
            spec = block_spec(int(x.shape[-1]), str(x.device), bool(wants_ft))
            if spec is not None:
                cnn_blocks = spec
        side_logit, ev_logit = symmetric_logits(members, x, mask, cnn_blocks,
                                                x_mirror=x_mirror)
        geo = flow = None
        free = None
        if use_geometry and frame_paths is not None and fps and collision_frame is not None:
            geo = geometric_side_score(frame_paths, frame_numbers, fps, collision_frame)
            flow = flow_side_score(frame_paths, frame_numbers, fps, collision_frame)
            if use_free_space:
                free = free_space_evasion(frame_paths, frame_numbers, fps, collision_frame,
                                          device=str(device or "cpu"))
        bias = evasion_bias(wants_ft) if evasion_bias_override is None \
            else float(evasion_bias_override)
        return {"entry_side": decide_side(side_logit, geo, flow),
                "evasion_space": decide_evasion(ev_logit, bias, free),
                "side_logit_sym": side_logit, "evasion_logit": ev_logit,
                "geo_score": geo, "flow_score": flow, "free_space": free}
    except Exception:                                   # noqa: BLE001
        return None


__all__ = ["SIDE_LABELS", "GEO_WEIGHT", "FLOW_WEIGHT", "EVASION_BIAS", "EVASION_BIAS_FT", "evasion_bias",
           "EVASION_MAX_NEIGHBOUR", "EVASION_MAX_SCENE", "free_space_evasion",
           "free_space_counts", "load_detector",
           "FLOW_PRE", "flow_side_score", "flow_side_score_from_grays",
           "mirror_features", "symmetric_logits", "block_spec",
           "BAND_ORDER_BAND", "BAND_ORDER_CHAN",
           "geometric_side_score", "geometric_side_score_from_grays", "decide_side", "decide_evasion", "override"]
