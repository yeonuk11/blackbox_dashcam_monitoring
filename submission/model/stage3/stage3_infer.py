"""Stage 3 inference: one (accel, steer) label per submitted sample.

Contract with ``inference.py``::

    load_model(model_dir, device=..., budget=..., deadline=...) -> handle
    predict_video(handle, path, meta=..., n_samples=..., mode=..., sec_per_item=...)
        -> (accel_labels, steer_labels), each exactly ``n_samples`` long

Three things this module is responsible for, in order of how much they cost when they go
wrong:

1. **The extractor must match the checkpoint.**  ``stage3_features`` reduces an optical-flow
   field to 70 numbers whose *scale* is the flow estimator's own: measured on the public
   OPEN_001.mp4, the same trained network labels the clip CONSTANT/DECELERATING from
   raft_large@512x384 features (its training front-end) and **STOPPED for all 600 samples**
   from DIS@384x288 features.  So the flow kind and working size are read back out of the
   checkpoint (``extra.flow`` / ``extra.size``) instead of being hard-coded here, and any
   time the extractor actually used is not the trained one the output is treated as
   suspect (see 3).

2. **The time budget decides the extractor, not the other way round.**  raft_large costs
   8.4 s/sample on this machine's CPU and ~7 ms/sample on its GPU; DIS costs ~30-50 ms/sample
   on the CPU and needs no GPU at all.  ``sec_per_item`` (from ``common.PerItemPacer``) picks
   the best rung of a quality ladder that still fits, the ladder is re-costed from the
   measured throughput after every video, and when even the cheapest rung does not fit the
   *sampling stride* grows -- features are computed for every s-th sample and held -- so the
   frame stays complete and valid no matter how little time is left.

3. **A single label for a whole clip is a scoring disaster.**  Stage 3 is macro-F1 over all
   four accel classes, so a collapsed clip drags the mean down for everyone.  When the
   features did not come from the trained front-end and the network still returns one class
   for >95% of the clip, the flow rule below takes over: measured on 60 held-out comma2k19
   segments it scores 0.478 (accel 0.497 / steer 0.433) against 0.218 for the constant
   CONSTANT+STRAIGHT answer.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import model_io

__all__ = ["load_model", "predict_video", "rule_labels", "FlowPlan"]

# --------------------------------------------------------------------------------------
# Feature indices used by the model-free rule. Resolved by name so a change to
# stage3_features.FEATURE_NAMES cannot silently point these at the wrong column.
# --------------------------------------------------------------------------------------
_SPEED_FEATURES = ("mag_p90_road", "sky_p90", "band_p90_3", "band_p90_4", "band_p90_5")
_STEER_FEATURES = (("grid_u_mean_r1c1", 0.7), ("horiz_mean_u", 0.3))

# Rule thresholds. Everything is expressed relative to the clip's own p90 flow magnitude
# because the absolute scale belongs to the flow estimator, not to the car: the same
# thresholds tuned on raft_large label 87% of a raft_small clip STOPPED when applied
# absolutely. `STOP_FLOOR` is the one absolute number left and only decides the degenerate
# "this clip never moves at all" case.
STOP_REL = 0.25          # |flow| below this fraction of the clip p90 counts as stopped
STOP_FLOOR = 0.15        # log1p px; below this p90 the whole clip is stopped
SLOPE_REL = 0.004        # d(speed)/dsample, in clip-p90 units, that separates ACCEL/DECEL
STEER_REL = 0.12         # horizontal flow, in clip-p90 units, that separates LEFT/RIGHT
SMOOTH_SPEED = 5         # 0.5 s boxcar on the speed proxy
SMOOTH_SLOPE = 13        # 1.3 s boxcar before differentiating (accel is a ~0.5 s property)

# Seconds per sample, measured on this machine 2026-08-27 (RTX 5090, 24 core) on the public
# 1164x874 clips. These are *end to end* -- decode included -- which is why the GPU rungs are
# not far below the CPU ones and why halving the working size barely helps: at this
# resolution libavcodec, not the flow network, is the bottleneck. The numbers only seed the
# ladder; `_Cost` replaces each with the rate actually observed after the first video, so a
# faster (or busier) evaluation box self-corrects within one item.
_COST_PRIOR = {
    ("raft_large", 512, 384, "cuda"): 0.060,
    ("raft_large", 384, 288, "cuda"): 0.050,
    ("raft_small", 512, 384, "cuda"): 0.046,
    ("raft_small", 384, 288, "cuda"): 0.040,
    ("dis", 512, 384, "cpu"): 0.070,
    ("dis", 384, 288, "cpu"): 0.040,
    ("dis", 256, 192, "cpu"): 0.025,
}
_CPU_RAFT_COST = 8.4     # raft_large @512x384 on CPU, measured: never affordable


@dataclass
class FlowPlan:
    kind: str            # "raft_large" | "raft_small" | "dis"
    width: int
    height: int
    device: str
    stride: int = 1      # compute one feature row per `stride` samples and hold the label

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def key(self) -> tuple:
        return (self.kind, self.width, self.height, self.device)

    def __str__(self) -> str:
        s = f"{self.kind}@{self.width}x{self.height}/{self.device}"
        return s if self.stride == 1 else f"{s} stride={self.stride}"


class _Cost:
    """Seconds per sample per plan, seeded from the priors and updated from reality.

    The priors are only ever *replaced* for a plan that actually ran, which used to make a
    stale prior self-fulfilling: the ensemble partner and the full-resolution rung are
    gated on ``cost(plan) * n_samples <= sec_per_item``, so a prior that is too pessimistic
    keeps its plan from ever running, and therefore from ever being measured.  Measured
    2026-08-28 under ``taskset -c 0-6`` on the public clips: dis@384x288 costs 0.0055 s per
    sample, not the 0.040 in the table (the table predates ``_dis_threads()``), and
    raft_large@512x384 costs 0.0093, not 0.060 -- so on any Stage 3 set bigger than ~70
    clips the partner was skipped on every single item while the box had 4x the budget it
    needed.  ``scale()`` closes that loop: whatever ratio the plans we *did* run show
    between reality and their prior is carried over to the plans we have not run yet.
    """

    def __init__(self) -> None:
        self.seen: dict[tuple, float] = {}

    def scale(self) -> float:
        """Geometric mean of observed/prior over every plan measured so far."""
        import math

        rs = [self.seen[k] / _COST_PRIOR[k] for k in self.seen
              if k in _COST_PRIOR and _COST_PRIOR[k] > 0 and self.seen[k] > 0]
        if not rs:
            return 1.0
        g = math.exp(sum(math.log(r) for r in rs) / len(rs))
        return float(min(5.0, max(0.2, g)))

    def get(self, plan: FlowPlan) -> float:
        if plan.key in self.seen:
            return self.seen[plan.key]
        if plan.kind.startswith("raft") and plan.device != "cuda":
            return _CPU_RAFT_COST          # never affordable; not worth calibrating
        return _COST_PRIOR.get(plan.key, 0.05) * self.scale()

    def update(self, plan: FlowPlan, seconds: float, n_rows: int) -> None:
        if n_rows <= 0 or seconds <= 0:
            return
        per = seconds / n_rows
        old = self.seen.get(plan.key)
        self.seen[plan.key] = per if old is None else 0.5 * (old + per)


def _log(msg: str) -> None:
    print(f"[stage3_infer] {msg}", file=sys.stderr, flush=True)


def _dis_threads() -> int:
    """OpenCV threads for the DIS flow front-end.

    ``stage3_features`` warns that DIS "collapses under thread oversubscription", which is
    true of the *training* path: that runs a pool of extractor processes, so each one must
    stay single-threaded.  Inference is the opposite shape -- one process, seven vCPUs, and
    DIS is 83% of the per-file cost -- so leaving it at one thread wastes six of them.
    ``DACON_DIS_THREADS`` overrides; the default is measured in
    ``scripts/bench_server_timing.py``.
    """
    import os

    try:
        n = int(os.environ.get("DACON_DIS_THREADS", "0"))
    except ValueError:
        n = 0
    if n > 0:
        return n
    return max(1, min(6, (os.cpu_count() or 1)))


# --------------------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------------------
def _find_checkpoint(model_dir) -> Path | None:
    """Stage 3's checkpoint and no other stage's. See ``model_io.find_checkpoint``.

    The RAFT weights that ride along in ``model/stage3/`` are excluded there by the
    ``<name>-<8 hex>.pth`` pretrained-release pattern, not by a "raft" substring: a Stage 3
    bundle is free to be named after the flow front-end it was trained on.
    """
    return model_io.find_checkpoint(model_dir, 3, who="stage3_infer")


AUX_NAMES = ("aux.pt", "stage3_aux.pt")


def _find_aux(model_dir) -> Path | None:
    """The optional second bundle of the two-front-end ensemble.

    ``model_io.find_checkpoint`` always answers with the *primary* (``best.pt`` is a
    preferred name and wins any tie), so the ensemble partner has to be addressed by an
    exact name.  Missing file, wrong stage directory, unreadable bundle -- every one of
    those simply means "run the single model", which is the pre-ensemble behaviour.
    """
    d = Path(model_dir)
    roots = [d] if d.is_dir() else [d.parent]
    sub = roots[0] / "stage3"
    if sub.is_dir():
        roots.append(sub)
    for root in roots:
        for name in AUX_NAMES:
            p = root / name
            if p.is_file():
                return p
    return None


def _load_bundle_defensively(path: Path, device: str, attempts: int = 3):
    """``torch.load`` a checkpoint that another process may be rewriting underneath us.

    Stage 3 is retrained continuously, and ``torch.save`` is not atomic: reading a
    half-written file raises (``UnpicklingError``, ``RuntimeError``, ``EOFError``, ...) or -
    worse - succeeds with a truncated state dict. Retrying after a second is enough because
    a 7 MB write finishes long inside that; a stable size across two looks is the signal.
    """
    ck = model_io.read_checkpoint(path, attempts=attempts, who="stage3_infer")
    if ck is None:
        _log(f"giving up on {path} -> running the flow rule instead")
        return None, None
    try:
        from stage3_model import Stage3Net

        a = ck["arch"]
        # Dilation is not a parameter shape, so a checkpoint trained with a different
        # dilation cycle would load *silently* into the default (1,2,4,8) stack and
        # compute a different function. Read it back from the bundle.
        net = Stage3Net(a["in_dim"], hidden=a["hidden"], layers=a["layers"],
                        dilations=a.get("dilations") or (1, 2, 4, 8),
                        feature_names=ck.get("feature_names"))
    except Exception as exc:
        _log(f"{path.name} does not describe a Stage3Net ({type(exc).__name__}: {exc}) "
             f"-> flow rule instead")
        return None, None
    ok, reason = model_io.load_into(net, ck, who="stage3_infer")
    if not ok:
        _log(f"{path.name} rejected: {reason} -> flow rule instead")
        return None, None
    net.to(device).eval()
    _log(f"{path.name}: {reason}")
    return net, ck


_RAFT_FILENAMES = {"raft_large": "raft_large_C_T_SKHT_V2-ff5fadd5.pth",
                   "raft_small": "raft_small_C_T_V2-01064c6d.pth"}


def _raft_weights(model_dir) -> dict[str, str]:
    """{model name: checkpoint path} for the RAFT weights shipped with the submission.

    Inside the zip the weights live in ``model/stage3/`` next to the Stage 3 bundle, which is
    not one of the directories ``stage3_features.default_weight_path`` searches, so resolve
    them here and pass the path explicitly. Keyed by model because the two files are not
    interchangeable -- loading raft_small's state dict into raft_large raises.
    """
    d = Path(model_dir)
    out: dict[str, str] = {}
    for kind, name in _RAFT_FILENAMES.items():
        for base in (d, d.parent, Path(__file__).resolve().parent):
            p = base / name
            if p.is_file():
                out[kind] = str(p)
                break
        if kind not in out:
            try:
                import stage3_features as SF

                found = SF.default_weight_path(kind)
                if found:
                    out[kind] = found
            except Exception:
                pass
    return out


def load_model(model_dir, device: str = "cuda", budget=None, deadline=None) -> dict:
    """Never raises: a missing or broken checkpoint degrades to the flow rule."""
    try:
        import torch

        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    except Exception:
        dev = "cpu"

    handle: dict = {"net": None, "ck": None, "device": dev, "model_dir": str(model_dir),
                    "cost": _Cost(), "extractors": {}, "raft_ok": dev == "cuda",
                    "train_plan": None, "raft_weights": _raft_weights(model_dir)}

    ckpt = _find_checkpoint(model_dir)
    if ckpt is None:
        _log(f"no Stage 3 checkpoint under {model_dir} -> flow rule only")
        return handle
    net, ck = _load_bundle_defensively(ckpt, dev)
    if net is None:
        return handle
    extra = ck.get("extra") or {}
    kind = str(extra.get("flow") or "raft_large")
    size = extra.get("size") or (512, 384)
    try:
        w, h = int(size[0]), int(size[1])
    except Exception:
        w, h = 512, 384
    handle["net"] = net
    handle["ck"] = ck
    try:
        from stage3_model import build_members

        handle["nets"] = build_members(ck, net, dev)
    except Exception as exc:
        _log(f"seed members unavailable ({type(exc).__name__}: {exc}) -> single net")
        handle["nets"] = [net]
    if len(handle["nets"]) > 1:
        _log(f"{ckpt.name}: averaging {len(handle['nets'])} seeds from the bundle")
    handle["train_plan"] = FlowPlan(kind, w, h, "cuda" if kind.startswith("raft") else "cpu")
    handle["decode"] = str(extra.get("decode") or "viterbi")
    _log(f"loaded {ckpt.name} ({extra.get('variant', '?')}, val={extra.get('val_score')}) "
         f"trained on {kind}@{w}x{h}; device={dev}, decode={handle['decode']}")

    # -- optional ensemble partner ------------------------------------------------------
    # Measured on the leave-one-vehicle-out split (train CIVIC, validate RAV4, public
    # example clips held out): DIS alone 0.7326, raft_large alone 0.7312, the two averaged
    # 0.7463 -- and 0.7507 once the prior-robust logit offsets are applied. The partner
    # costs a second flow front-end, so it only runs when the pacer says the item can
    # afford both; everything below degrades to the single model rather than raising.
    handle["aux"] = None
    try:
        ap = _find_aux(model_dir)
        if ap is not None and ckpt is not None and ap.resolve() == ckpt.resolve():
            # ``find_checkpoint`` falls back to "newest loose .pt" when best.pt is absent, so
            # a bundle shipped with only aux.pt would load the same file twice and pay for a
            # second identical flow front-end to average a model with itself.
            _log(f"ensemble partner {ap.name} is the primary checkpoint -> single model")
            ap = None
        if ap is not None and dev == "cuda":
            anet, ack = _load_bundle_defensively(ap, dev)
            if anet is not None:
                aex = ack.get("extra") or {}
                akind = str(aex.get("flow") or "raft_large")
                asize = aex.get("size") or (512, 384)
                aw = float((extra.get("aux_weight") if extra.get("aux_weight") is not None
                            else 1.0 - float(extra.get("ens_weight_dis", 0.5))))
                try:
                    from stage3_model import build_members

                    anets = build_members(ack, anet, dev)
                except Exception:
                    anets = [anet]
                handle["aux"] = {
                    "net": anet, "nets": anets, "ck": ack,
                    "weight": max(0.0, min(1.0, aw)),
                    "plan": FlowPlan(akind, int(asize[0]), int(asize[1]),
                                     "cuda" if akind.startswith("raft") else "cpu"),
                }
                _log(f"ensemble partner {ap.name}: {akind}@{asize[0]}x{asize[1]} "
                     f"weight {handle['aux']['weight']:.2f}, "
                     f"{len(handle['aux']['nets'])} seed(s)")
        elif ap is not None:
            _log(f"ensemble partner {ap.name} ignored: no CUDA, its front-end would cost "
                 f"{_CPU_RAFT_COST} s/sample on the CPU")
    except Exception as exc:
        _log(f"ensemble partner unavailable ({type(exc).__name__}: {exc}) -> single model")
        handle["aux"] = None
    return handle


# --------------------------------------------------------------------------------------
# Model-free flow rule -- the floor under every code path
# --------------------------------------------------------------------------------------
def _idx(names: Sequence[str]) -> list[int]:
    import stage3_features as SF

    lookup = {n: i for i, n in enumerate(SF.FEATURE_NAMES)}
    return [lookup[n] for n in names if n in lookup]


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size < 2:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")[: x.size]


def _prior_labels(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The answer for a clip that yielded no usable features at all.

    An unreadable file is not a stopped car, and the tempting answer -- one constant label --
    is the worst available: macro-F1 divides by all four accel classes whether or not they
    appear. ``common.s3_default_labels`` lays the training prior down in contiguous blocks,
    which keeps every class represented; the same filler backs ``sanitize_stage3``.
    """
    a, s = common.s3_default_labels(n)
    return (np.array([common.ACCEL_TO_IDX[x] for x in a], dtype=np.int64),
            np.array([common.STEER_TO_IDX[x] for x in s], dtype=np.int64))


def rule_labels(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(accel_idx, steer_idx) straight from the flow features, no network involved.

    Scored on 60 comma2k19 segments with raft_large features: accel macro-F1 0.497,
    steer 0.433, Stage 3 total 0.478 -- against 0.218 for constant CONSTANT/STRAIGHT.
    """
    n = int(feats.shape[0])
    accel = np.full(n, common.ACCEL_TO_IDX["CONSTANT"], dtype=np.int64)
    steer = np.full(n, common.STEER_TO_IDX["STRAIGHT"], dtype=np.int64)
    if n == 0:
        return accel, steer
    try:
        i_speed = _idx(_SPEED_FEATURES)
        i_steer = [(_idx([n_])[0], w) for n_, w in _STEER_FEATURES if _idx([n_])]
        v = feats[:, i_speed].mean(1) if i_speed else np.zeros(n, np.float32)
        p90 = float(np.quantile(v, 0.90)) if n > 1 else float(v[0] if n else 0.0)
        scale = max(p90, STOP_FLOOR)
        if p90 < STOP_FLOOR:                    # the clip never moves
            accel[:] = common.ACCEL_TO_IDX["STOPPED"]
            return accel, steer
        stopped = _smooth(v, SMOOTH_SPEED) < max(0.5 * STOP_FLOOR, STOP_REL * p90)
        slope = np.gradient(_smooth(v, SMOOTH_SLOPE)) / scale if n > 2 else np.zeros(n)
        accel[slope > SLOPE_REL] = common.ACCEL_TO_IDX["ACCELERATING"]
        accel[slope < -SLOPE_REL] = common.ACCEL_TO_IDX["DECELERATING"]
        accel[stopped] = common.ACCEL_TO_IDX["STOPPED"]
        if i_steer:
            u = sum(w * feats[:, i] for i, w in i_steer)
            u = _smooth(np.asarray(u, dtype=np.float64), SMOOTH_SLOPE) / scale
            steer[u > STEER_REL] = common.STEER_TO_IDX["LEFT"]      # camera yaws left ->
            steer[u < -STEER_REL] = common.STEER_TO_IDX["RIGHT"]    # scene streams right
    except Exception:
        # Swallowing here used to return the CONSTANT/STRAIGHT arrays this function starts
        # from -- the single-class answer the whole module exists to avoid, and the one the
        # caller trusts this path never to produce ("the rule ... never collapses"). Hand
        # back the prior mix instead, which is what every other give-up path returns.
        return _prior_labels(n)
    return accel, steer


# --------------------------------------------------------------------------------------
# Plan selection
# --------------------------------------------------------------------------------------
def _ladder(handle: dict) -> list[FlowPlan]:
    """Extractors from best to cheapest, deduplicated, GPU-only rungs dropped on CPU."""
    dev = handle["device"]
    gpu = dev == "cuda" and handle.get("raft_ok", False)
    out: list[FlowPlan] = []
    train = handle.get("train_plan")
    if train is not None and (gpu or not train.kind.startswith("raft")):
        out.append(FlowPlan(train.kind, train.width, train.height,
                            "cuda" if train.kind.startswith("raft") else "cpu"))
    if gpu and (train is None or train.kind.startswith("raft")):
        # Only a RAFT-trained primary may degrade onto raft_small: the features of the two
        # families are not interchangeable.  Measured on the leave-one-vehicle-out val, the
        # DIS-trained model scores 0.7322 on its own features and 0.3220 on raft_small ones
        # -- below even the model-free flow rule -- and 92% of the rows come out CONSTANT,
        # which slips under the 95% ``_collapsed`` guard.  For a DIS primary raft_small also
        # buys nothing: its prior cost equals dis@384x288's, so it was never a cheaper rung.
        out.append(FlowPlan("raft_small", 384, 288, "cuda"))
    out += [FlowPlan("dis", 384, 288, "cpu"), FlowPlan("dis", 256, 192, "cpu")]
    seen, uniq = set(), []
    for p in out:
        if p.key not in seen:
            seen.add(p.key)
            uniq.append(p)
    return uniq


def _choose_plan(handle: dict, n_samples: int, mode: str, sec_per_item: float | None
                 ) -> FlowPlan:
    ladder = _ladder(handle)
    start = 1 if (mode == "fast" and len(ladder) > 1) else 0
    cost = handle["cost"]
    if sec_per_item is None or sec_per_item <= 0:
        return ladder[start]
    for plan in ladder[start:]:
        if cost.get(plan) * n_samples <= sec_per_item:
            return plan
    # Nothing fits at full sampling: keep the cheapest extractor and thin the grid instead.
    plan = ladder[-1]
    need = cost.get(plan) * n_samples
    stride = int(np.ceil(need / max(1e-3, sec_per_item)))
    plan.stride = max(1, min(stride, max(1, n_samples // 8)))
    return plan


def _build_extractor(handle: dict, plan: FlowPlan):
    """Cached construction; a RAFT failure disables the GPU rungs for the whole run."""
    ex = handle["extractors"].get(plan.key)
    if ex is not None:
        return ex
    import stage3_features as SF

    kw: dict = {"size": plan.size, "batch_size": 8, "backend": "auto"}
    if plan.kind.startswith("raft"):
        try:
            ex = SF.FlowExtractor(device=plan.device, model=plan.kind,
                                  weights_path=(handle.get("raft_weights") or {}).get(plan.kind),
                                  allow_download=False, **kw)
        except Exception as exc:
            _log(f"{plan.kind} unavailable ({type(exc).__name__}: {exc}) -> DIS from here on")
            handle["raft_ok"] = False
            return None
    else:
        ex = SF.DISFlowExtractor(cv_threads=_dis_threads(), feat_device=handle["device"],
                                 **kw)
    handle["extractors"][plan.key] = ex
    return ex


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def _net_logprobs(net, feats: np.ndarray, dev: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-class log-softmax -- the only scale on which members can be mixed.

    ``net`` may be one Stage3Net or a list of same-architecture seeds stored in the same
    bundle (see ``stage3_model.save_bundle(members=...)``), in which case their log-softmax
    is averaged.  Measured on the honest leave-one-vehicle-out split (2026-08-29), going
    from one net per front-end to three raises Stage 3 from 0.7758 to 0.7811 -- a clip
    bootstrap puts that at +0.0052 [+0.0028, +0.0077] -- and costs 4.8 ms per extra seed
    per 600-sample clip against ~20 s of optical flow and decode, so the average is always
    taken when the weights are there.
    """
    import torch

    nets = net if isinstance(net, (list, tuple)) else [net]
    x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))[None].to(dev)
    la = ls = None
    with torch.inference_mode():
        for n in nets:
            out = n(x)
            a = torch.log_softmax(out["accel"][0].float(), -1)
            s = torch.log_softmax(out["steer"][0].float(), -1)
            la = a if la is None else la + a
            ls = s if ls is None else ls + s
    k = max(1, len(nets))
    return (la.cpu().numpy().astype(np.float64) / k,
            ls.cpu().numpy().astype(np.float64) / k)


def _decode_with_model(handle: dict, feats: np.ndarray,
                       aux_lp: tuple[np.ndarray, np.ndarray] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    from stage3_model import viterbi_decode

    net, ck = handle.get("nets") or handle["net"], handle["ck"]
    la, ls = _net_logprobs(net, feats, handle["device"])
    if aux_lp is not None:
        w = float((handle.get("aux") or {}).get("weight", 0.5))
        n = min(len(la), len(aux_lp[0]))
        la = (1.0 - w) * la[:n] + w * aux_lp[0][:n]
        ls = (1.0 - w) * ls[:n] + w * aux_lp[1][:n]
    ba, bs, src = _bias_vectors(ck)
    if not handle.get("_bias_logged"):
        handle["_bias_logged"] = True
        _log(f"class bias in force ({src}): accel {np.round(ba, 3).tolist()} | "
             f"steer {np.round(bs, 3).tolist()}")
    la = la + ba
    ls = ls + bs
    if str(handle.get("decode") or "viterbi") == "argmax":
        # Measured on the honest leave-one-vehicle-out split: Viterbi is worth +0.002 on a
        # single model and -0.005 on the ensemble, whose averaged posteriors are already
        # smooth. The bundle says which decoder it was validated with.
        return la.argmax(1), ls.argmax(1)
    dwell = ck.get("min_dwell") or {}
    a = viterbi_decode(la, np.asarray(ck["trans_accel"], dtype=np.float64),
                       min_dwell=max(1, int(dwell.get("accel", 2))))
    s = viterbi_decode(ls, np.asarray(ck["trans_steer"], dtype=np.float64),
                       min_dwell=max(1, int(dwell.get("steer", 2))))
    return a, s


# --------------------------------------------------------------------------------------
# Diagnostic switch: which class-bias operating point ships.
# --------------------------------------------------------------------------------------
# ``None`` = use whatever ``bias_accel`` / ``bias_steer`` the checkpoint carries (normal
# behaviour).  A ``(accel4, steer3)`` pair overrides it for the whole run.
#
# Why this exists.  The only clean transfer measurement we have is submission 1 -> 3, and
# it is worth stating with the configurations that were actually shipped rather than with
# whichever ablation rows happened to be at hand (re-measured 2026-08-29 on the pinned LOVO
# pair models/stage3/exp/lovo_{dis,raft_large}_sj2.5.pt, 373 val clips / 223 431 samples):
#
#   submission 1   DIS alone, its own bias, viterbi                 LOVO 0.7291
#   submission 3/4 0.4*DIS + 0.6*raft_large, sub-4 bias, argmax     LOVO 0.7619
#   LOVO predicts +0.0328; the private score moved 0.67646 -> 0.69402, i.e. +0.01756.
#
# So the honest transfer ratio for that step is ~0.54, not 1.0.  (An earlier version of this
# comment quoted +0.0181 from 0.7326 -> 0.7507; those are a *different* primary checkpoint
# -- lovo_dis_sj2.5_s2 -- with a zero-bias baseline and a mean-of-priors bias that never
# shipped, so the two endpoints do not bracket a shipped change.)
#
# Submission 3 -> 5 then changed exactly two things: 1 net -> 3 nets per front-end (+0.0053
# LOVO) and this bias vector.  It did NOT change the decoder: the submission-3/4 bundle
# already carries extra["decode"] == "argmax" (analysis/stage3_backup_20260829/best.pt), so
# viterbi -> argmax belongs to the 1 -> 3 step, not to 3 -> 5.  The private score moved
# -0.00073.
#
# The case *against* the submission-5 bias is only that its steer component was tuned partly
# on the five public example clips, a surface known not to transfer.  Every local surface
# that is not public-5 prefers it: on the 7-prior LOVO panel it beats the sub-4 bias on all
# seven priors (+0.0018 .. +0.0138, mean +0.0078, scripts/s3_bias_sensitivity.py), and on the
# public five's dense 10 Hz key it calls RIGHT 396 times against a truth of 371 where the
# sub-4 bias calls it 167 times.  Reverting is therefore a *diagnostic*, not an improvement:
# expect it to cost ~0.004-0.008 of Stage 3 unless the public-5 tuning really did poison it.
#
# ``BIAS_SUB4`` is the submission-3/4 operating point (private Stage 3 0.69402, fitted only
# on the training fold across a panel of assumed priors).  Setting BIAS_OVERRIDE = BIAS_SUB4
# is the single-variable diagnostic: everything else stays at submission 5.  Because the
# leaderboard reports S2 and S3 separately, it does not need a submission of its own -- ride
# it along with the next Stage 2 change and both stages stay identified.
BIAS_SUB4 = ((0.5, 0.5, 0.0, -0.9), (0.6, -0.3, 0.3))
# ``BIAS_LEVERD`` is submission 5's bias, read out of that bundle's own best.pt. The
# leaderboard identified both halves of the day-3 step separately, holding one fixed:
#   sub3/4 nets + SUB4 bias = 0.69402 | sub5 nets + SUB4 bias = 0.68740  -> nets -0.00662
#   sub5 nets + SUB4 bias  = 0.68740 | sub5 nets + leverD bias = 0.69329 -> bias +0.00589
# So the 3+3-seed bundle is the dud and the leverD bias is the keeper. This ships the
# winning half of each: the sub3/4 checkpoints (restored into models/stage3) under the
# leverD bias. Predicted S3 ~0.6999 if the two effects add.
BIAS_LEVERD = ((0.4, 0.2, 0.0, -0.9), (0.7, 0.0, 2.2))
BIAS_OVERRIDE: tuple[Sequence[float], Sequence[float]] | None = None


def _bias_vectors(ck: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(accel bias, steer bias, where it came from). Never raises."""
    if BIAS_OVERRIDE is not None:
        try:
            ba = np.asarray(BIAS_OVERRIDE[0], dtype=np.float64).reshape(-1)
            bs = np.asarray(BIAS_OVERRIDE[1], dtype=np.float64).reshape(-1)
            if ba.size == len(common.ACCEL_LABELS) and bs.size == len(common.STEER_LABELS):
                return ba, bs, "BIAS_OVERRIDE"
            _log(f"BIAS_OVERRIDE has the wrong shape ({ba.size},{bs.size}) -> checkpoint bias")
        except Exception as exc:
            _log(f"BIAS_OVERRIDE unusable ({type(exc).__name__}: {exc}) -> checkpoint bias")
    return (np.asarray(ck.get("bias_accel", 0.0), dtype=np.float64),
            np.asarray(ck.get("bias_steer", 0.0), dtype=np.float64), "checkpoint")


def _collapsed(labels: np.ndarray, frac: float = 0.95) -> bool:
    if labels.size == 0:
        return True
    return float(np.bincount(labels).max()) / labels.size >= frac


def _expand(idx: np.ndarray, stride: int, n: int) -> np.ndarray:
    """Undo a strided feature grid: sample k takes the label of row ``k // stride``."""
    if stride <= 1 and idx.size >= n:
        return idx[:n]
    src = np.minimum(np.arange(n) // max(1, stride), max(0, idx.size - 1))
    return idx[src] if idx.size else np.zeros(n, dtype=np.int64)


def predict_video(handle, path, meta=None, n_samples: int = 0, mode: str = "full",
                  sec_per_item: float | None = None) -> tuple[list[str], list[str]]:
    """Exactly ``n_samples`` (accel, steer) labels. Raises only if numpy itself fails."""
    import stage3_features as SF

    n = max(1, int(n_samples or (common.s3_sample_count(meta) if meta is not None else 1)))
    handle = handle if isinstance(handle, dict) else {"net": None, "ck": None, "device": "cpu",
                                                      "cost": _Cost(), "extractors": {},
                                                      "raft_ok": False, "train_plan": None,
                                                      "raft_weights": {}}
    plan = _choose_plan(handle, n, mode, sec_per_item)
    n_rows = int(np.ceil(n / plan.stride))
    fps = common.S3_HZ
    if meta is not None:
        try:
            _, fps = SF.sample_grid(meta, n)
        except Exception:
            fps = common.S3_HZ
    hz = common.S3_HZ / plan.stride           # a strided grid is simply a slower sample rate

    feats = np.zeros((n_rows, SF.FEATURE_DIM), dtype=np.float32)
    t0 = time.monotonic()
    ex = _build_extractor(handle, plan)
    if ex is None:                            # RAFT vanished: re-plan on the DIS ladder
        plan = _choose_plan(handle, n, "fast", sec_per_item)
        n_rows = int(np.ceil(n / plan.stride))
        hz = common.S3_HZ / plan.stride
        ex = _build_extractor(handle, plan)
    if ex is not None:
        try:
            feats = ex.extract(path, n_rows, fps, hz=hz)
        except Exception as exc:
            _log(f"{Path(str(path)).name}: {plan} extraction failed "
                 f"({type(exc).__name__}: {exc}) -> zero features")
    handle["cost"].update(plan, time.monotonic() - t0, n_rows)

    if not np.any(feats):
        a, s = _prior_labels(n)
        _log(f"{Path(str(path)).name}: no features recovered -> prior-mix labels for "
             f"{n} samples")
        return ([common.ACCEL_LABELS[int(i)] for i in a],
                [common.STEER_LABELS[int(i)] for i in s])

    a_rule, s_rule = rule_labels(feats)
    a, s = a_rule, s_rule
    used = "rule"

    # -- ensemble partner, if the item can afford a second flow front-end ---------------
    aux_lp = None
    aux = handle.get("aux")
    if (aux is not None and handle.get("net") is not None and mode == "full"
            and plan.stride == 1):
        cost = handle["cost"]
        need = (cost.get(plan) + cost.get(aux["plan"])) * n
        if sec_per_item is None or need <= sec_per_item:
            t1 = time.monotonic()
            try:
                aex = _build_extractor(handle, aux["plan"])
                if aex is None:
                    # RAFT could not be built at all (missing weights, no CUDA): it will not
                    # appear later either, so stop paying the constructor on every clip.
                    _log("ensemble partner front-end unavailable -> single model from here on")
                    handle["aux"] = None
                else:
                    af = aex.extract(path, n_rows, fps, hz=hz)
                    handle["cost"].update(aux["plan"], time.monotonic() - t1, n_rows)
                    if np.any(af):
                        aux_lp = _net_logprobs(aux.get("nets") or aux["net"], af,
                                               handle["device"])
            except Exception as exc:
                _log(f"{Path(str(path)).name}: ensemble partner failed "
                     f"({type(exc).__name__}: {exc}) -> primary model alone")
                aux_lp = None
        else:
            _log(f"{Path(str(path)).name}: skipping the ensemble partner, {need:.1f}s needed "
                 f"vs {sec_per_item:.1f}s of budget for this item")

    if handle.get("net") is not None:
        try:
            a_m, s_m = _decode_with_model(handle, feats, aux_lp)
            used = "model+aux" if aux_lp is not None else "model"
            train = handle.get("train_plan")
            matched = train is not None and (plan.kind, plan.width, plan.height) == \
                (train.kind, train.width, train.height)
            if not matched and (_collapsed(a_m) and not _collapsed(a_rule)):
                # Out-of-distribution features + a one-class answer: exactly the failure the
                # module docstring describes. The rule is worse on average but never collapses.
                _log(f"{Path(str(path)).name}: model collapsed to a single accel class on "
                     f"{plan} features (trained on {train}) -> flow rule")
                used = "rule(model-collapsed)"
            else:
                a, s = a_m, s_m
        except Exception as exc:
            _log(f"{Path(str(path)).name}: model decode failed "
                 f"({type(exc).__name__}: {exc}) -> flow rule")

    a = _expand(np.asarray(a, dtype=np.int64), plan.stride, n)
    s = _expand(np.asarray(s, dtype=np.int64), plan.stride, n)
    dt = time.monotonic() - t0
    _log(f"{Path(str(path)).name}: {n} samples, {plan}, {used}, {dt:.1f}s "
         f"({dt / max(1, n) * 1000:.1f} ms/sample)")
    return ([common.ACCEL_LABELS[int(i)] for i in a],
            [common.STEER_LABELS[int(i)] for i in s])
