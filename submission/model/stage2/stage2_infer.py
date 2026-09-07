"""Stage 2 inference: collision / entry frame, entry side, evasion space, from a JPEG folder.

Contract with ``inference.py``::

    load_model(model_dir, device=..., budget=..., deadline=...) -> handle
    predict_folder(handle, folder, frames=..., frame_numbers=..., mode=..., sec_per_item=...)
        -> {"collision_frame": int, "entry_frame": int, "evasion_space": 0|1,
            "entry_side": "LEFT"|"RIGHT", "evasion_score": float, "side_score": float}

There is no trained Stage 2 model yet, so everything here is image analysis with nothing but
OpenCV -- no detector weights to ship, nothing to download.  What it exploits is the one
thing the organizers guaranteed: **every evaluation clip contains a collision, and the
dashcam car is a party to it** (talk board tb 417202).  An impact the camera itself takes is
the largest single-frame change in the whole clip, so the collision is a peak-finding problem
rather than a detection problem.

The two categorical fields return a *score* as well as a label.  Macro-F1 counts both
categories of each, which makes a constant answer the worst possible one -- 0.333 against a
50/50 truth where any balanced split scores ~0.5 -- and the measured priors really are
balanced (1048 AI Hub 597 clips: entry_side LEFT 50.3% / RIGHT 49.7%).  ``inference.py``
USED TO re-threshold these scores at their median over the whole evaluation set; that is a
statistic over the evaluation set and rule 4 forbids it, so ``_rebalance_stage2`` is now a
deliberate no-op and the scores are diagnostics only.  Anti-collapse is a per-file job now:
the trained path gets it from the mirrored TTA pass (``stage2_decode.run_nets`` averages the
plain and mirrored side logits, which removes the mirror-invariant "say LEFT" offset --
measured 82% LEFT -> 55% LEFT on 400 real long clips), not from any cross-file cut.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s2common as common
import s2model_io as model_io

__all__ = ["load_model", "predict_folder", "collision_curve", "entry_side_score",
           "evasion_score"]

# Analysis geometry, in fractions of the frame. The ego lane is the calibrated dashcam
# trapezoid used by stage2_propose (vanishing point at (0.50 W, 0.55 H); half lane width at
# the image bottom = 1.30 * (H - vp_y), clipped to [0.16, 0.48] W -- on a flat road a fixed
# lateral offset projects to (X / camera_height) * (y - y_vp) whatever the focal length).
VP_XY = (0.50, 0.55)
LANE_HALF_OVER_H, LANE_HALF_MIN, LANE_HALF_MAX = 1.30, 0.16, 0.48
HOOD_Y = 0.88                # below this the ego bonnet fills the frame: ignore
SKY_Y = 0.28                 # above this there is nothing but sky
SIDE_WINDOW_SEC = 1.5        # how much run-up decides which side the victim came from
COARSE_TARGET = 320          # frames read in the coarse pass, whatever the folder length
# Degradation ladder. The pacer in inference.py drops to "fast" when it projects an overrun
# and to "min" when the projection is dire; the cost of each rung is measured on the AI Hub
# held-out split (scripts/bench_stage2_degrade.py), never assumed.
_COARSE_BY_MODE = {"full": COARSE_TARGET, "fast": COARSE_TARGET // 2, "min": 96}
REFINE_SEC = 1.2             # full-rate window around the coarse peak
IMPACT_SEC = 0.4             # width of the box filter that defines "the impact"


def _log(msg: str) -> None:
    print(f"[stage2_infer] {msg}", file=sys.stderr, flush=True)


# How a trained Stage 2 reaches this module.  ``scripts/train_stage2.py`` and
# ``src/stage2_model.py`` are owned by another pipeline and did not exist when this was
# written, so nothing here assumes their API: the module is a *router*.  If a stage-2
# checkpoint appears in ``model_dir`` and ``stage2_model`` can turn it into something that
# predicts, that path runs; otherwise the OpenCV heuristic below runs, unchanged.  The
# adapters are tried in order and the first that works wins:
#
#   1. ``stage2_model.load_model(model_dir, device=...)``  -- their own loader, used whole
#   2. ``stage2_model.load_bundle(path, device=...)``       -- (net, ck) like stage3_model
#   3. ``stage2_model.Stage2Net(**ck["cfg"])`` / ``build_model(ck["cfg"])`` + state dict
#
# Prediction then delegates to ``stage2_model.predict_folder`` (or ``predict_one`` /
# ``predict_item``).  inference.py probes ``stage2_infer`` before ``stage2_model``, so
# without this routing a trained model would sit in the zip and never be called.
# ``stage2_decode`` is the decode + inference glue this module prefers when it is present:
# same features, same checkpoints, but the full 10 Hz grid instead of a 384-sample cap, a
# mirrored TTA pass that reuses the optical flow, a tuned decode window, and a
# coarse-to-fine fallback for when the pacer cannot afford the full grid.  It is tried
# first and ``stage2_model``'s own predictor is the fallback, so removing the file is a
# complete rollback to what submission #3 ran.
_DECODE_MODULE = "stage2_decode"
_LOADERS = ("load_model", "load_stage_model")
_BUNDLES = ("load_bundle", "load_checkpoint")
_BUILDERS = ("build_model", "make_model", "Stage2Net", "Stage2Model")
_PREDICTORS = ("predict_folder", "predict_one", "predict_item")
_DELEGATE_ERROR_LIMIT = 3        # consecutive bad answers before the heuristic takes over


def _import_stage2_model():
    try:
        import stage2_model

        return stage2_model
    except Exception as exc:                      # noqa: BLE001 - absence is the normal case
        _log(f"stage2_model unavailable ({type(exc).__name__}: {exc}) -> heuristic")
        return None


def _own(mod, names):
    """First callable among ``names`` that ``mod`` defines itself (not a re-export)."""
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn) and getattr(fn, "__module__", None) == getattr(mod, "__name__", None):
            return n, fn
    for n in names:                               # a deliberate re-export is still usable
        fn = getattr(mod, n, None)
        if callable(fn):
            return n, fn
    return None, None


def _build_trained(mod, ckpt, device: str):
    """(inner_handle, how) from whatever ``stage2_model`` exposes, or (None, reason)."""
    name, loader = _own(mod, _LOADERS)
    if loader is not None:
        try:
            h = model_io.call_filtered(loader, str(ckpt.parent), device=device)
            if h is not None:
                return h, f"stage2_model.{name}()"
        except Exception as exc:                  # noqa: BLE001
            _log(f"stage2_model.{name}() raised {type(exc).__name__}: {exc}")

    name, bundle = _own(mod, _BUNDLES)
    if bundle is not None:
        try:
            got = model_io.call_filtered(bundle, str(ckpt), device=device)
            net = got[0] if isinstance(got, tuple) else got
            if net is not None:
                return net, f"stage2_model.{name}()"
        except Exception as exc:                  # noqa: BLE001
            _log(f"stage2_model.{name}() raised {type(exc).__name__}: {exc}")

    ck = model_io.read_checkpoint(ckpt, who="stage2_infer")
    if ck is None:
        return None, "checkpoint unreadable"
    name, builder = _own(mod, _BUILDERS)
    if builder is None:
        return None, f"stage2_model exposes none of {_LOADERS + _BUNDLES + _BUILDERS}"
    cfg = ck.get("cfg") or ck.get("config") or ck.get("arch") or {}
    try:
        net = builder(**cfg) if isinstance(cfg, dict) else builder(cfg)
    except Exception as exc:                      # noqa: BLE001
        return None, f"stage2_model.{name}(cfg) raised {type(exc).__name__}: {exc}"
    ok, reason = model_io.load_into(net, ck, who="stage2_infer")
    if not ok:
        return None, reason
    try:
        net.to(device).eval()
    except Exception:                             # noqa: BLE001 - not a torch module
        pass
    return net, f"stage2_model.{name} + validated state dict ({reason})"


def _decode_predictor():
    """``(name, fn)`` of ``stage2_decode.predict_folder``, or ``(None, None)``.

    Absence is not an error -- it just means the shipped ``stage2_model.predict_folder``
    runs instead -- but it has to be *said*, because the difference between the two is
    worth several points of Stage 2 and is otherwise invisible in the server log.
    """
    try:
        # A literal import, not importlib: build_submission's closure pass walks the AST for
        # import statements, and a module reached only through importlib would be left out
        # of the zip -- where the fallback is silent and costs the whole gain.
        import stage2_decode

        fn = getattr(stage2_decode, "predict_folder", None)
        if callable(fn):
            return f"{_DECODE_MODULE}.predict_folder", fn
        _log(f"{_DECODE_MODULE} has no predict_folder -> stage2_model's own predictor")
    except Exception as exc:                      # noqa: BLE001
        _log(f"{_DECODE_MODULE} unavailable ({type(exc).__name__}: {exc}) "
             f"-> stage2_model's own predictor")
    return None, None


def load_model(model_dir, device: str = "cpu", budget=None, deadline=None) -> dict:
    """Trained Stage 2 if one is in ``model_dir``, else the heuristic. Never raises."""
    try:
        import cv2

        cv2.setNumThreads(1)       # 7 vCPU shared with everything else
    except Exception:
        pass
    handle: dict = {"model_dir": str(model_dir), "net": None, "delegate": None,
                    "device": device, "errors": 0}

    ckpt = model_io.find_checkpoint(model_dir, 2, who="stage2_infer")
    if ckpt is None:
        _log("no Stage 2 checkpoint -> OpenCV heuristic")
        return handle
    mod = _import_stage2_model()
    if mod is None:
        return handle
    inner, how = _build_trained(mod, ckpt, device)
    if inner is None:
        _log(f"{ckpt.name} unusable ({how}) -> OpenCV heuristic")
        return handle
    pname, pred = _own(mod, _PREDICTORS)
    if pname:
        pname = f"stage2_model.{pname}"
    dname, dpred = _decode_predictor()
    if dpred is not None:
        pname, pred = dname, dpred
    if pred is None:
        # The weights loaded but there is no way to run them: stage2_model defines the
        # network and not the inference glue (feature extraction + forward + decode). Name
        # the gap, because from the outside this is indistinguishable from "no model".
        have = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
        _log(f"stage2_model has weights but none of {_PREDICTORS}; it exposes {have[:8]} "
             f"-> OpenCV heuristic. Add one of {_PREDICTORS} taking "
             f"(net, folder, frames=, frame_numbers=, mode=, sec_per_item=) and returning "
             f"collision_frame/entry_frame/entry_side/evasion_space to activate it.")
        return handle
    handle["net"] = inner
    handle["delegate"] = pred
    _log(f"loaded {ckpt.name} via {how}; predicting with {pname}()")
    # The delegate may still be unable to run -- its feature extractor needs pretrained
    # backbone weights that live outside the checkpoint. Ask it to prove it can build them
    # now, so a missing weight file is one loud line here instead of an invisible
    # per-folder demotion to the heuristic (which is how Stage 2 scored 0.238 once).
    pre = getattr(mod, "preflight", None)
    if callable(pre):
        try:
            # ``net=`` is filtered out for a preflight that does not take it. stage2_model's
            # does: it is the only place that can notice a fine-tuned checkpoint shipped
            # without its backbone file, which loads and runs and is silently wrong.
            where = model_io.call_filtered(pre, device=device, net=inner)
            _log(f"preflight ok: {pname}() can build its feature extractor ({where})")
        except Exception as exc:                  # noqa: BLE001
            _log(f"PREFLIGHT FAILED ({type(exc).__name__}: {exc}). stage2_model.{pname}() "
                 f"cannot build its feature extractor, so every folder will fall back to "
                 f"the OpenCV heuristic. Ship the backbone weights beside the checkpoint "
                 f"(model/stage2/) or under models/pretrained/.")
    return handle


def _entry_policy(out: dict | None, numbers: Sequence[int], n_frames: int) -> dict | None:
    """Apply ``stage2_entry``'s ``entry_frame`` policy to whichever delegate answered.

    THE choke point for the entry column.  It has to live here, not in a decoder, because
    ``load_model`` picks the delegate at run time: ``stage2_decode.predict_folder`` when it
    can build the net's feature layout, ``stage2_model.predict_folder`` when it cannot, and
    the OpenCV heuristic below when neither loads.  A hook inside one of them is silently
    dead whenever the other runs -- which is exactly what happened: ``stage2_model`` grew
    an ``apply_entry_policy`` call on 2026-08-29 and ten real 1250-frame folders came back
    BYTE-IDENTICAL under ``ENTRY_POLICY`` "head" and "collision", because
    ``stage2_decode.predict_folder`` was answering all ten
    (``analysis/stage2_entry_v2/e2e_bench.json``).

    Pure integer arithmetic on the folder's own frame NUMBERS: no I/O, no model, no
    cross-file state (rule 4), and it cannot raise.  ``entry <= collision`` is preserved by
    ``apply_entry_policy`` and the answer is snapped onto a frame the folder contains, so
    the result still satisfies everything ``_sane`` just checked.  With ``stage2_entry``
    absent -- it is pulled into the zip by ``build_submission``'s import closure, but the
    guard costs nothing -- the delegate's own answer stands, which is submissions #1-#4.
    """
    if not isinstance(out, dict) or "collision_frame" not in out or "entry_frame" not in out:
        return out
    try:
        import stage2_entry

        c, e = int(out["collision_frame"]), int(out["entry_frame"])
        e2 = int(stage2_entry.apply_entry_policy(
            c, e, common.s2_assumed_fps(max(int(n_frames), 1)),
            frame_numbers=list(numbers) or None))
        if e2 != e and (not numbers or min(numbers) <= e2 <= max(numbers)) and e2 <= c:
            out = dict(out)
            out["entry_frame"] = e2
            # The delegate already logged its own ``ent=``; say when the submitted value is
            # a different one, or a probe is invisible in the server log.
            _log(f"entry policy '{stage2_entry.ENTRY_POLICY}': entry {e} -> {e2} "
                 f"(collision {c})")
    except Exception:                              # noqa: BLE001
        pass
    return out


def _sane(out, numbers: Sequence[int]) -> dict | None:
    """A delegate's answer, or None if it is not something worth submitting.

    A model that returns NaN, a frame number the folder does not contain, or a label outside
    the two-value vocabulary is worse than the heuristic, and ``sanitize_stage2`` would only
    turn it into the *prior* -- so catch it here where the heuristic is still available.
    """
    if not isinstance(out, dict) or not out:
        return None
    lo, hi = (min(numbers), max(numbers)) if numbers else (0, 0)
    got = {}
    for key in ("collision_frame", "entry_frame"):
        try:
            v = float(out[key])
        except Exception:                          # noqa: BLE001
            return None
        if not np.isfinite(v) or not (lo <= v <= hi):
            return None
        got[key] = int(round(v))
    if got["entry_frame"] > got["collision_frame"]:
        got["entry_frame"] = got["collision_frame"]
    side = str(out.get("entry_side", "")).upper().strip()
    if side not in common.SIDE_LABELS:
        return None
    try:
        ev = int(out.get("evasion_space"))
    except Exception:                              # noqa: BLE001
        return None
    if ev not in (0, 1):
        return None
    got["entry_side"] = side
    got["evasion_space"] = ev
    # Carry the ranking scores through when the model provides them. They are diagnostics
    # only now: inference.py used to re-cut both categorical fields at the median of these
    # scores across the evaluation set, which is a cross-file statistic and rule 4 forbids
    # it (`inference._rebalance_stage2` is a deliberate no-op). Anti-collapse is handled
    # per file instead, inside stage2_side.
    got["side_score"] = float(out.get("side_score", 1.0 if side == "RIGHT" else -1.0))
    got["evasion_score"] = float(out.get("evasion_score", float(ev)))
    return got


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def _read(path, reduce: int = 8, gray: bool = True):
    import cv2

    flag = {(8, True): cv2.IMREAD_REDUCED_GRAYSCALE_8,
            (4, True): cv2.IMREAD_REDUCED_GRAYSCALE_4,
            (2, True): cv2.IMREAD_REDUCED_GRAYSCALE_2,
            (8, False): cv2.IMREAD_REDUCED_COLOR_8,
            (4, False): cv2.IMREAD_REDUCED_COLOR_4,
            (2, False): cv2.IMREAD_REDUCED_COLOR_2}.get(
        (reduce, gray), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    # REDUCED_* asks libjpeg for a DCT-scaled decode, which is several times cheaper than
    # decoding full resolution and resizing -- the difference between 3 ms and 12 ms a frame
    # over the ~1250 frames of an evaluation folder.
    return cv2.imread(str(path), flag)


def _lane(width: float, height: float) -> tuple[float, float, float]:
    """(vp_x, vp_y, half_width_at_bottom) in pixels."""
    vx, vy = VP_XY[0] * width, VP_XY[1] * height
    half = float(np.clip(LANE_HALF_OVER_H * (height - vy),
                         LANE_HALF_MIN * width, LANE_HALF_MAX * width))
    return vx, vy, half


# --------------------------------------------------------------------------------------
# Collision
# --------------------------------------------------------------------------------------
def collision_curve(images: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame change energy, robustly standardised. ``images`` are consecutive greys.

    The statistic is the mean absolute inter-frame difference over the drivable part of the
    frame (sky and bonnet excluded), standardised by its own median and MAD so that a night
    clip, a rainy clip and a bright motorway clip all land on the same scale. An ego-involved
    impact is the largest excursion in this curve: the camera is struck, so the whole image
    moves at once, which no amount of ordinary driving reproduces.
    """
    n = len(images)
    out = np.zeros(max(0, n - 1), dtype=np.float32)
    if n < 2:
        return out
    h, w = images[0].shape[:2]
    y0, y1 = int(SKY_Y * h), max(int(SKY_Y * h) + 1, int(HOOD_Y * h))
    prev = images[0][y0:y1].astype(np.float32)
    for i in range(1, n):
        cur = images[i][y0:y1].astype(np.float32)
        if cur.shape != prev.shape:
            prev = cur
            continue
        out[i - 1] = float(np.abs(cur - prev).mean())
        prev = cur
    med = float(np.median(out))
    mad = float(np.median(np.abs(out - med))) + 1e-6
    return (out - med) / (1.4826 * mad)


def _peak(curve: np.ndarray, box: int) -> int:
    """Index of the strongest ``box``-wide excursion (the impact is not one frame wide)."""
    if curve.size == 0:
        return 0
    k = max(1, min(int(box), curve.size))
    pos = np.clip(curve, 0.0, None)
    acc = np.convolve(pos, np.ones(k), mode="same")
    return int(np.argmax(acc))


# --------------------------------------------------------------------------------------
# Entry side
# --------------------------------------------------------------------------------------
def entry_side_score(images: Sequence[np.ndarray]) -> float:
    """>0 means the victim came from screen RIGHT, <0 from screen LEFT.

    A vehicle cutting into the ego lane is the one large object whose image motion does not
    match the forward flow of the static scene, and it arrives on one side. Differencing
    consecutive frames and comparing the change energy of the left and right thirds -- with
    the centre column, sky and bonnet masked out, because those are dominated by ego motion --
    puts the excess on the side the intruder came from. The value is normalised to [-1, 1] so
    it can be compared across clips.
    """
    if len(images) < 2:
        return 0.0
    h, w = images[0].shape[:2]
    y0, y1 = int(SKY_Y * h), max(int(SKY_Y * h) + 1, int(HOOD_Y * h))
    xl0, xl1 = 0, int(0.36 * w)
    xr0, xr1 = int(0.64 * w), w
    left = right = 0.0
    prev = images[0].astype(np.float32)
    for i in range(1, len(images)):
        cur = images[i].astype(np.float32)
        if cur.shape != prev.shape:
            prev = cur
            continue
        d = np.abs(cur[y0:y1] - prev[y0:y1])
        left += float(d[:, xl0:xl1].mean())
        right += float(d[:, xr0:xr1].mean())
        prev = cur
    tot = left + right
    return 0.0 if tot <= 1e-6 else float((right - left) / tot)


# --------------------------------------------------------------------------------------
# Evasion space
# --------------------------------------------------------------------------------------
def evasion_score(image: np.ndarray) -> float:
    """How much room there was to swerve, in [0, 1]; higher means more room.

    Both neighbouring lanes are evaluated on a lookahead row 45% of the way from the
    vanishing point to the image bottom -- roughly where a swerve would actually go, and far
    enough forward that a neighbouring lane is still inside the frame. A lane counts as free
    when its centre at that row is on screen and its road patch looks like the ego lane's
    (similar mean colour, similar edge density). That similarity test is the cheap stand-in
    for drivable-area segmentation: it is what rejects grass, kerbs, medians, barriers and
    parked-car clutter without a single learned parameter.
    """
    import cv2

    if image is None or image.ndim != 3:
        return 0.0
    h, w = image.shape[:2]
    vx, vy, half = _lane(float(w), float(h))
    y = vy + 0.45 * (h - vy)
    t = (y - vy) / max(1e-6, h - vy)
    half_y = half * max(0.0, t)
    if half_y < 4:
        return 0.0

    def patch(cx: float) -> tuple[np.ndarray, float] | None:
        x0, x1 = int(cx - 0.8 * half_y), int(cx + 0.8 * half_y)
        y0, y1 = int(y - 0.035 * h), int(y + 0.035 * h)
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 < 6 or y1 - y0 < 4:
            return None
        p = image[y0:y1, x0:x1]
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return p.reshape(-1, 3).mean(0), float(np.hypot(gx, gy).mean())

    ego = patch(vx)
    if ego is None:
        return 0.0
    best = 0.0
    for k in (-1, 1):
        cx = vx + k * 2.0 * half_y
        if not (0.02 * w < cx < 0.98 * w):
            continue                       # the ego car is already at the road boundary
        nb = patch(cx)
        if nb is None:
            continue
        colour = float(np.linalg.norm(nb[0] - ego[0]))
        edges = nb[1] / max(1e-3, ego[1])
        # 1.0 for a lane that looks exactly like the ego lane, decaying to 0 by 42/255 of
        # colour distance or 2.5x the edge density -- the thresholds stage2_propose uses.
        s = max(0.0, 1.0 - colour / 42.0) * max(0.0, 1.0 - max(0.0, edges - 1.0) / 1.5)
        best = max(best, float(np.clip(s, 0.0, 1.0)))
    return best


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def predict_folder(handle, folder, frames=None, frame_numbers=None, mode: str = "full",
                   sec_per_item: float | None = None) -> dict:
    t_start = time.monotonic()
    handle = handle if isinstance(handle, dict) else {}
    frames = [Path(p) for p in (frames or [])]
    numbers = [int(x) for x in (frame_numbers or [])]
    if not frames:
        frames = common.list_frames(folder)
        numbers = [common.frame_number(p) for p in frames]
    if not numbers:
        numbers = list(range(len(frames)))
    n = len(frames)

    delegate = handle.get("delegate")
    if delegate is not None:
        try:
            out = _sane(model_io.call_filtered(delegate, handle.get("net"), folder,
                                               frames=frames, frame_numbers=numbers,
                                               mode=mode, sec_per_item=sec_per_item),
                        numbers)
        except Exception as exc:                   # noqa: BLE001
            _log(f"stage2_model delegate raised {type(exc).__name__}: {exc}")
            out = None
        if out is not None:
            handle["errors"] = 0
            return _entry_policy(out, numbers, n)
        handle["errors"] = int(handle.get("errors", 0)) + 1
        if handle["errors"] >= _DELEGATE_ERROR_LIMIT:
            _log(f"delegate failed {handle['errors']}x in a row -> heuristic for the rest "
                 f"of the run")
            handle["delegate"] = None
    if n == 0:
        return {}
    fps = common.s2_assumed_fps(n)
    if n < 4:
        return {**common.s2_defaults(Path(str(folder)).name, numbers, n),
                "evasion_score": 0.5, "side_score": 0.0}

    # --- coarse pass: a fixed frame budget, so a 1250-frame folder costs the same as a 50 ---
    budget_frames = _COARSE_BY_MODE.get(mode, COARSE_TARGET)
    if sec_per_item and sec_per_item > 0:
        # ~3 ms per DCT-scaled 1/8 read measured on 1920x1080 JPEGs; leave a third of the
        # per-item budget for the refine pass and the two categorical heuristics.
        budget_frames = int(min(budget_frames, max(24, 0.66 * sec_per_item / 0.003)))
    # The refine pass costs ~1.2 s * fps full-rate reads on top of the coarse pass. Drop it
    # once the per-item budget is tight enough that the coarse pass itself is being starved:
    # a peak located to within a couple of frames is already inside the +-0.3 s tolerance.
    do_refine = mode != "min" and budget_frames > _COARSE_BY_MODE["min"]
    stride = max(1, int(np.ceil(n / max(8, budget_frames))))
    coarse_pos = list(range(0, n, stride))
    coarse = [_read(frames[i], 8, True) for i in coarse_pos]
    coarse_pos = [p for p, img in zip(coarse_pos, coarse) if img is not None]
    coarse = [img for img in coarse if img is not None]
    if len(coarse) < 3:
        return {**common.s2_defaults(Path(str(folder)).name, numbers, n),
                "evasion_score": 0.5, "side_score": 0.0}
    curve = collision_curve(coarse)
    ci = _peak(curve, max(1, int(round(IMPACT_SEC * fps / stride))))
    pos = coarse_pos[min(ci + 1, len(coarse_pos) - 1)]

    # --- refine at full rate inside a short window around the coarse peak ---
    # "min" skips it: the refine window is 1.2 s of full-rate reads, about a third of the
    # per-folder cost, and it only ever moves the peak by a few frames.
    if stride > 1 and do_refine:
        half = int(round(0.5 * REFINE_SEC * fps))
        lo, hi = max(0, pos - half), min(n, pos + half + 1)
        fine = [_read(frames[i], 8, True) for i in range(lo, hi)]
        keep = [(i, img) for i, img in zip(range(lo, hi), fine) if img is not None]
        if len(keep) >= 3:
            fcurve = collision_curve([img for _, img in keep])
            fi = _peak(fcurve, max(1, int(round(IMPACT_SEC * fps))))
            pos = keep[min(fi + 1, len(keep) - 1)][0]

    collision = numbers[min(pos, n - 1)]
    lead = int(round(common.S2_ENTRY_LEAD_SEC * fps))
    entry_pos = max(0, pos - lead)
    entry = numbers[entry_pos]

    # --- the two categorical fields, from the run-up to the impact ---
    side_lo = max(0, pos - int(round(SIDE_WINDOW_SEC * fps)))
    side_step = max(1, (pos - side_lo) // 12)
    side_imgs = [img for img in (_read(frames[i], 8, True)
                                 for i in range(side_lo, max(side_lo + 1, pos), side_step))
                 if img is not None]
    s_score = entry_side_score(side_imgs)
    colour = _read(frames[min(pos, n - 1)], 4, False)
    e_score = evasion_score(colour)

    # The heuristic answers under the same entry policy as the trained delegates, so a
    # leaderboard probe is not diluted by whichever folders fell back to it. Applied BEFORE
    # the log line: a server log that prints a different entry_frame than the one submitted
    # is how a probe gets misread.
    entry = int(min(entry, collision))
    try:
        import stage2_entry

        entry = int(stage2_entry.apply_entry_policy(int(collision), entry, fps,
                                                    frame_numbers=numbers or None))
    except Exception:                              # noqa: BLE001
        pass
    dt = time.monotonic() - t_start
    _log(f"{Path(str(folder)).name}: {n} frames, mode={mode}, stride {stride}, "
         f"refine={do_refine}, "
         f"collision={collision} entry={entry} side={s_score:+.3f} evasion={e_score:.3f} "
         f"in {dt:.2f}s")
    return {
        "collision_frame": int(collision),
        "entry_frame": int(min(entry, collision)),
        # Provisional labels: inference.py re-thresholds both at the median of the score over
        # the whole evaluation set, because macro-F1 punishes a collapsed split hardest.
        "entry_side": "RIGHT" if s_score > 0 else "LEFT",
        "evasion_space": 1 if e_score >= 0.5 else 0,
        "side_score": float(s_score),
        "evasion_score": float(e_score),
    }
