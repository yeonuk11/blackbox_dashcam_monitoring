"""Shared conventions for all three stages.

Everything in this module must be importable in the offline evaluation server
(torch 2.8.0+cu128, torchvision 0.23.0, opencv-python-headless 4.10, av>=15,<17,
numpy 1.26.4, pandas 2.2.2, scipy 1.15.3, scikit-learn 1.5.2) with **no network**.

Design rules enforced here (from the competition evaluation page):
  * A raised exception in any predict_stageN aborts the WHOLE submission (no partial
    score), so every public entry point returns a valid DataFrame no matter what.
  * Rows with NaN / negative / out-of-range frames or labels outside the allowed set
    score zero, so `sanitize_stageN` clamps and validates before returning.
  * import + model load time counts against the 60 min budget -> heavy imports are lazy.
"""
from __future__ import annotations

import os
import sys
import time
import math
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------------------
# Offline / determinism environment. Must run before torch, timm, transformers,
# ultralytics are imported anywhere in the process.
# --------------------------------------------------------------------------------------
def set_offline_env(tmp_root: str | None = None) -> str:
    import tempfile

    tmp = tmp_root or os.path.join(tempfile.gettempdir(), "dacon_cache")
    for sub in ("hf", "torch", "yolo", "mpl"):
        try:
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        except Exception:
            pass
    env = {
        # Hugging Face
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HOME": os.path.join(tmp, "hf"),
        # torch.hub / torchvision weight cache (we never download, but keep it writable)
        "TORCH_HOME": os.path.join(tmp, "torch"),
        # ultralytics: is_online() short-circuits to False, disabling telemetry,
        # AutoUpdate (pip) and release-asset downloads.
        "YOLO_OFFLINE": "true",
        "YOLO_CONFIG_DIR": os.path.join(tmp, "yolo"),
        "MPLCONFIGDIR": os.path.join(tmp, "mpl"),
        # determinism for cuBLAS GEMMs
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # keep BLAS from oversubscribing the 7 vCPUs when DataLoader workers are used
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return tmp


def seed_everything(seed: int = 20260826) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Label vocabularies -- exact strings the scorer accepts.
# --------------------------------------------------------------------------------------
S1_LABELS = ("ORIGINAL", "RERECORDED")
ACCEL_LABELS = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_LABELS = ("LEFT", "STRAIGHT", "RIGHT")
SIDE_LABELS = ("LEFT", "RIGHT")

ACCEL_TO_IDX = {v: i for i, v in enumerate(ACCEL_LABELS)}
STEER_TO_IDX = {v: i for i, v in enumerate(STEER_LABELS)}

VIDEO_EXT = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv", ".flv",
    ".mpg", ".mpeg", ".ts", ".webm", ".hevc", ".h265", ".h264", ".mts", ".m2ts",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Stage 3 sampling rate fixed by the organizers ("Stage 3 영상은 10Hz 기준으로 구성").
S3_HZ = 10.0
# Fallback fps when the container reports something impossible. The five public
# examples decode at 20 fps even though the container claims 479 fps.
S3_FALLBACK_FPS = 20.0
GENERIC_FALLBACK_FPS = 30.0

# --------------------------------------------------------------------------------------
# Stage 2 priors -- the constants a stage with no model still has to answer with.
#
# Measured on 2026-08-27 from the labels actually on disk, scoring each candidate constant
# with the official Acc@+-0.3s rather than eyeballing a median (scripts print the table in
# analysis/stage2_prior.md).  Per-corpus optimum of `collision_frame = r * (n_frames - 1)`:
#
#   corpus                 n   median frames   best r   acc@best r   acc@r=0.70
#   CCD ego-involved     801              50    0.664       0.4794       0.3833
#   Nexar w/ contact     400            1207    0.497       0.3225       0.0000
#   Nexar all positive   750            1206    0.497       0.3013       0.0013
#   AI Hub 597           501             150    0.514       0.3912       0.1058
#   equal-weight pooled                         0.497       0.2433       0.1226
#
# Every accident corpus cuts its clips around the event, and three of the four optima sit
# within 0.02 of the midpoint, so 0.50 it is -- it doubles the expected fallback accuracy
# against the 0.70 that was hard-coded here before.  The private folders are ~1250 frames
# (organizers' notice: frame_000000..frame_001249), i.e. Nexar-shaped, where 0.70 scores 0.
S2_COLLISION_REL = 0.50
# entry_frame = collision_frame - S2_ENTRY_LEAD_SEC * fps.  From 540 AI Hub 597 clips whose
# isObjectB anchor track enters the ego-lane trapezoid before impact: the lead is heavily
# right-skewed (median 0.60 s, p25 0.20 s, p75 1.67 s) and the constant maximising
# Acc@+-0.3s is 0.30 s (50.9% vs 28.3% at the median 0.6 s and 17.8% at 1.0 s).
S2_ENTRY_LEAD_SEC = 0.30
# Stage 2 ships frames, never a container, so there is no fps to read.  The public examples
# are 50-frame/5 s CCD clips (10 fps); the private folders are ~1250 frames and every long
# dashcam corpus on disk (Nexar 1206 fr, AI Hub 150 fr) runs at 30.  Split on length.
S2_SHORT_CLIP_FRAMES = 80
S2_ASSUMED_FPS_SHORT = 10.0
S2_ASSUMED_FPS_LONG = 30.0


def s2_assumed_fps(n_frames: int) -> float:
    return S2_ASSUMED_FPS_SHORT if int(n_frames) <= S2_SHORT_CLIP_FRAMES else S2_ASSUMED_FPS_LONG


# --------------------------------------------------------------------------------------
# Time budget: a single monotonic clock shared by all stages.
# --------------------------------------------------------------------------------------
@dataclass
class Budget:
    """Global deadline manager.

    ``total`` is the wall-clock limit of the whole submission (60 min on the server).
    Each stage gets a share; if an earlier stage overruns, later stages shrink
    automatically because every deadline is computed from the same ``t0``.
    """

    total: float = 60 * 60.0
    reserve: float = 5 * 60.0          # never spend the last N seconds
    share: dict = field(default_factory=lambda: {"stage1": 0.17, "stage2": 0.38, "stage3": 0.45})
    t0: float = field(default_factory=time.monotonic)
    _stage_start: dict = field(default_factory=dict)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def remaining(self) -> float:
        return max(0.0, self.total - self.reserve - self.elapsed())

    def start_stage(self, stage: str) -> float:
        """Begin a stage and return its deadline as an absolute monotonic time."""
        self._stage_start[stage] = time.monotonic()
        budget = self.remaining() * self.share.get(stage, 0.33) / max(
            1e-6, sum(self.share.get(s, 0.0) for s in self.share if s not in self._stage_start or s == stage)
        )
        # Never hand a stage more than what is actually left.
        budget = min(budget, self.remaining())
        return time.monotonic() + budget

    def stage_left(self, deadline: float) -> float:
        return max(0.0, min(deadline - time.monotonic(), self.remaining()))

    def expired(self, deadline: float) -> bool:
        return self.stage_left(deadline) <= 0.0


class PerItemPacer:
    """Extrapolates per-item cost and tells the caller when to degrade or bail out.

    Usage::

        pacer = PerItemPacer(len(items), deadline, budget)
        for it in items:
            mode = pacer.mode()           # "full" | "fast" | "skip"
            if mode == "skip":
                emit_default(it); continue
            ...
            pacer.tick()
    """

    def __init__(self, n_items: int, deadline: float, budget: Budget, warmup: int = 3):
        self.n = max(1, int(n_items))
        self.deadline = deadline
        self.budget = budget
        self.warmup = warmup
        self.done = 0
        self.t_start = time.monotonic()

    def _per_item(self) -> float:
        if self.done == 0:
            return 0.0
        return (time.monotonic() - self.t_start) / self.done

    def projected_overrun(self) -> float:
        """Seconds by which the current pace would overshoot the deadline (<=0 is fine)."""
        if self.done < self.warmup:
            return 0.0
        left_items = self.n - self.done
        need = left_items * self._per_item()
        return need - self.budget.stage_left(self.deadline)

    def per_item_budget(self) -> float:
        """Seconds each remaining item may spend if all of them are to fit.

        This is what a stage module needs in order to *choose* how to work -- Stage 3 picks
        RAFT vs DIS and the sampling resolution from it -- and it is meaningful from the
        very first item, unlike ``mode()``, which has to wait for the warm-up samples.
        """
        left_items = max(1, self.n - self.done)
        return self.budget.stage_left(self.deadline) / left_items

    def mode(self) -> str:
        left = self.budget.stage_left(self.deadline)
        if left <= 0:
            return "skip"
        if self.done < self.warmup:
            return "full"
        per = self._per_item()
        left_items = self.n - self.done
        if per * left_items <= left:
            return "full"
        # Degraded mode is assumed ~2.5x cheaper; if even that does not fit, skip.
        if per * left_items / 2.5 <= left:
            return "fast"
        return "skip" if per > left else "fast"

    def rate(self) -> float:
        """Items per second achieved so far (0 before the first tick)."""
        dt = time.monotonic() - self.t_start
        return self.done / dt if dt > 0 and self.done else 0.0

    def tick(self) -> None:
        self.done += 1


# --------------------------------------------------------------------------------------
# Evaluation-directory layout.
#
# The competition publishes two different trees and has not said which one the server
# builds (talk-board question tb 417208, unanswered as of 2026-08-27):
#
#   data tab               detail sheet test_data_structure.csv
#   data/stage1/videos/    data/stage1/
#   data/stage2/images/ID/ data/stage2/ID/
#   data/stage3/videos/    data/stage3/
#
# and ``predict_stageN(data_dir, ...)`` may itself be handed either ``.../stageN`` or the
# ``data/`` above it.  Guessing wrong costs the entire stage, so resolve by *looking*: try
# each plausible root and keep the first one that actually holds this stage's items.
# --------------------------------------------------------------------------------------
_STAGE_SUBDIRS = {1: ("videos", "video", "clips"),
                  2: ("images", "image", "frames", "imgs"),
                  3: ("videos", "video", "clips")}
_LAYOUT_MAX_DEPTH = 3


def _shallow_video_count(p: Path) -> int:
    try:
        return sum(1 for c in p.iterdir()
                   if c.is_file() and not c.name.startswith(".")
                   and c.suffix.lower() in VIDEO_EXT)
    except Exception:
        return 0


def _has_images(p: Path) -> bool:
    try:
        return any(c.is_file() and c.suffix.lower() in IMAGE_EXT for c in p.iterdir())
    except Exception:
        return False


def _shallow_frame_dir_count(p: Path) -> int:
    try:
        return sum(1 for c in p.iterdir()
                   if c.is_dir() and not c.name.startswith(".") and _has_images(c))
    except Exception:
        return 0


def _layout_candidates(data_dir: Path, stage: int) -> list[Path]:
    subs = _STAGE_SUBDIRS.get(int(stage), ())
    stage_dir = f"stage{int(stage)}"
    roots = [data_dir]
    # ``data_dir`` may be the parent that still contains stage1/ stage2/ stage3/, and the
    # baseline notebook occasionally passes the zip root, which contains data/.
    roots += [data_dir / stage_dir, data_dir / "data" / stage_dir, data_dir / "test" / stage_dir]
    out: list[Path] = []
    for r in roots:
        out += [r / s for s in subs]
        out.append(r)
    seen, uniq = set(), []
    for p in out:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def resolve_stage_dir(data_dir: str | Path, stage: int) -> Path:
    """Directory that *directly* holds stage ``stage``'s items, whatever the layout.

    Stage 1/3 look for video files one level down, Stage 2 for sub-folders that hold JPEGs.
    A candidate only wins if it really contains items, so an empty ``videos/`` next to the
    real files never shadows them.  Falls back to the deepest-first scan and finally to
    ``data_dir`` itself, which ``list_videos`` still searches recursively.
    """
    data_dir = Path(data_dir)
    count = _shallow_frame_dir_count if int(stage) == 2 else _shallow_video_count
    for cand in _layout_candidates(data_dir, stage):
        try:
            if cand.is_dir() and count(cand) > 0:
                return cand
        except Exception:
            continue
    # Nothing matched the known shapes: walk a bounded depth and take the shallowest
    # directory that holds items, so an unexpected extra level still resolves.
    best: tuple[int, Path] | None = None
    try:
        for cand in [data_dir, *sorted(p for p in data_dir.rglob("*") if p.is_dir())]:
            depth = len(cand.relative_to(data_dir).parts)
            if depth > _LAYOUT_MAX_DEPTH:
                continue
            if count(cand) > 0 and (best is None or depth < best[0]):
                best = (depth, cand)
    except Exception:
        pass
    if best is not None:
        return best[1]
    for cand in _layout_candidates(data_dir, stage):
        if cand.is_dir():
            return cand
    return data_dir


# --------------------------------------------------------------------------------------
# File discovery. IDs are file stems (stage 1/3) or folder names (stage 2).
# --------------------------------------------------------------------------------------
def list_videos(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in VIDEO_EXT:
            out.append(p)
    if not out:  # unknown extensions: probe with PyAV
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and _probe_openable(p):
                out.append(p)
    return out


def _probe_openable(path: Path) -> bool:
    try:
        import av

        with av.open(str(path)) as c:
            return len(c.streams.video) > 0
    except Exception:
        return False


def list_frame_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def frame_number(path: Path) -> int:
    """Frame index encoded in the filename (``frame_000123.jpg`` -> 123).

    The competition requires submitting the ORIGINAL frame number from the filename,
    never a re-indexed position.
    """
    import re

    m = re.search(r"(\d+)\s*$", path.stem)
    return int(m.group(1)) if m else 0


def list_frames(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    return sorted(files, key=frame_number)


# --------------------------------------------------------------------------------------
# Video metadata: fps that survives broken containers.
# --------------------------------------------------------------------------------------
@dataclass
class VideoMeta:
    path: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    n_frames: int = 0
    duration: float = 0.0
    fps_source: str = "unknown"
    ok: bool = False


def _rate_to_float(r) -> float:
    try:
        return float(r) if r is not None else 0.0
    except Exception:
        return 0.0


def _fps_from_pts(stream, packets_pts: list[int], durations: list[int]) -> tuple[float, str]:
    """Recover the real frame rate from packet timestamps.

    Container-level rates cannot be trusted: the Stage 3 example clips are raw HEVC
    elementary streams remuxed into mp4, where only the first ~50 packets carry a
    correct 60000-tick (= 0.05 s) step and the rest collapse to 1-2 tick increments.
    ffmpeg then reports average_rate 479.8 and *guessed_rate 40* -- the latter is
    inside any plausible range yet is exactly 2x wrong, which would halve the number
    of Stage 3 rows. Taking the median step over the well-formed prefix recovers the
    true 20 fps, and also gives 10 fps for the well-muxed Stage 1/2 clips.
    """
    import numpy as np

    tb = _rate_to_float(stream.time_base)
    if tb <= 0:
        return 0.0, ""
    head = np.asarray(packets_pts[:64], dtype=np.int64)
    if head.size >= 6:
        d = np.diff(np.sort(head))
        d = d[d > 0]
        if d.size >= 4:
            step = float(np.median(d))
            fps = 1.0 / (step * tb)
            if 1.0 <= fps <= 120.0:
                return fps, "pts-median"
    # Second opinion: the modal packet duration among plausible values.
    dur = np.asarray([x for x in durations[:512] if x and x > 0], dtype=np.int64)
    if dur.size >= 8:
        vals, cnt = np.unique(dur, return_counts=True)
        for i in np.argsort(-cnt):
            fps = 1.0 / (float(vals[i]) * tb)
            if 1.0 <= fps <= 120.0:
                return fps, "pkt-duration"
    return 0.0, ""


def probe_video(path: str | Path, fallback_fps: float = GENERIC_FALLBACK_FPS,
                count_frames: bool = True) -> VideoMeta:
    """Robust metadata probe.

    Priority for fps: packet-PTS median -> modal packet duration -> average_rate ->
    guessed_rate -> base_rate -> caller fallback. Frame count comes from demuxing
    (cheap, no decode) rather than the stream header, which is often absent or wrong.
    Duration is always derived as n_frames / fps, never from the container.
    """
    path = Path(path)
    meta = VideoMeta(path=path)
    try:
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            meta.width, meta.height = int(s.width or 0), int(s.height or 0)
            pts: list[int] = []
            durs: list[int] = []
            n_pkt = 0
            if count_frames:
                for pkt in c.demux(s):
                    if pkt.size is None or pkt.size <= 0:
                        continue
                    n_pkt += 1
                    if pkt.pts is not None:
                        pts.append(int(pkt.pts))
                        durs.append(int(pkt.duration) if pkt.duration else 0)
            else:
                for pkt in c.demux(s):
                    if pkt.size and pkt.pts is not None:
                        pts.append(int(pkt.pts))
                        durs.append(int(pkt.duration) if pkt.duration else 0)
                    if len(pts) >= 64:
                        break
                n_pkt = int(s.frames or 0)

            fps, src = _fps_from_pts(s, pts, durs)
            # A bare elementary stream (.hevc/.h265/.h264) carries no timestamps at
            # all; ffmpeg then invents 25 fps, which is worse than the caller's
            # domain fallback, so do not consult the container rates for those.
            raw_stream = path.suffix.lower() in {".hevc", ".h265", ".h264", ".265", ".264"}
            if fps <= 0 and not raw_stream:
                for name, v in (("average_rate", _rate_to_float(s.average_rate)),
                                ("guessed_rate", _rate_to_float(s.guessed_rate)),
                                ("base_rate", _rate_to_float(s.base_rate))):
                    if 1.0 <= v <= 120.0:
                        fps, src = v, name
                        break
            meta.fps, meta.fps_source = fps, src
            meta.n_frames = n_pkt if n_pkt > 0 else int(s.frames or 0)
            meta.ok = meta.width > 0 and meta.height > 0
    except Exception:
        pass

    if meta.n_frames <= 0:
        meta.n_frames = count_video_frames(path)
    if meta.fps <= 0:
        meta.fps, meta.fps_source = float(fallback_fps), "fallback"
    if meta.n_frames > 0 and meta.fps > 0:
        meta.duration = meta.n_frames / meta.fps
    return meta


def count_video_frames(path: str | Path) -> int:
    """Exact frame count by demuxing packets (cheap: no decode)."""
    try:
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            return sum(1 for p in c.demux(s) if p.size and p.pts is not None)
    except Exception:
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            n = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                n += 1
            cap.release()
            return n
        except Exception:
            return 0


def decode_frames(path: str | Path, indices: Sequence[int] | None = None,
                  size: tuple[int, int] | None = None, threads: str = "AUTO",
                  gray: bool = False):
    """Decode frames by index (never by timestamp).

    ``indices`` must be sorted ascending; missing frames reuse the previous decoded
    frame so a corrupt stream never shortens the output. ``size`` is (w, h) and is
    applied by libswscale during ``reformat`` -- much cheaper than decoding full-res
    and resizing afterwards.
    Yields ``(index, ndarray HxWx3 uint8 RGB)`` (or HxW when ``gray``).
    """
    import numpy as np

    want = None if indices is None else sorted({int(i) for i in indices if i >= 0})
    fmt = "gray" if gray else "rgb24"
    try:
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            with contextlib.suppress(Exception):
                s.thread_type = threads
                s.thread_count = 0
            last = None
            wi = 0
            for i, frame in enumerate(c.decode(s)):
                if want is not None:
                    if wi >= len(want):
                        break
                    if i < want[wi]:
                        continue
                try:
                    if size is not None:
                        arr = frame.reformat(width=size[0], height=size[1], format=fmt).to_ndarray()
                    else:
                        arr = frame.reformat(format=fmt).to_ndarray()
                    last = arr
                except Exception:
                    arr = last
                    if arr is None:
                        continue
                if want is None:
                    yield i, arr
                else:
                    # emit once per requested index (duplicates allowed if want repeats)
                    while wi < len(want) and want[wi] == i:
                        yield want[wi], arr
                        wi += 1
    except Exception:
        # OpenCV fallback (its FFmpeg build differs; used only if PyAV cannot open).
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            i, wi, last = 0, 0, None
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if want is None or (wi < len(want) and i >= want[wi]):
                    if size is not None:
                        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
                    last = arr
                    if want is None:
                        yield i, arr
                    else:
                        while wi < len(want) and want[wi] == i:
                            yield want[wi], arr
                            wi += 1
                i += 1
                if want is not None and wi >= len(want):
                    break
            cap.release()
        except Exception:
            return


# fps sources that were *measured* from the stream rather than read out of a container
# header, and are therefore trustworthy enough to contradict the announced 10 Hz.
_MEASURED_FPS_SOURCES = ("pts-median", "pkt-duration")


def s3_sample_count(meta: VideoMeta) -> int:
    """Number of Stage 3 output rows for one evaluation video.

    The organizers answered on the talk board (tb 417199, 2026-08-26): "비공개 Stage 3
    평가영상은 10Hz로 구성되며, sample_index 개수는 디코딩된 프레임 수와 동일합니다."
    -- the private clips are 10 fps, and there is one row per decoded frame.

    Note what that sentence is: a rule *plus its premise*. At 10 fps "one row per decoded
    frame" and "one row per 0.1 s" are the same number, so the rule tells us nothing about
    which of the two the scorer actually builds -- and the premise is false for every
    example they shipped. All five public clips decode at a measured 20 fps (packet-PTS
    median; 1197-1201 frames over ~60 s), and the organizers' own answer key for them,
    baseline/data/stage3/labels.csv, runs sample_index 0..540 in steps of 60 against
    frame_index 0..1080 in steps of 120: a **600**-sample grid, exactly half the frame
    count. Submitting one row per frame there would double every clip.

    So: one row per 0.1 s of video, which *equals* the decoded frame count whenever the
    announced 10 Hz holds and therefore never contradicts the answer -- but only when the
    frame rate was measured from the stream (never from a container header, which lies:
    these files claim average_rate 479.8). Anything less certain falls back to the
    organizers' literal wording.
    """
    n = int(meta.n_frames)
    if n > 0:
        if (meta.fps_source in _MEASURED_FPS_SOURCES and meta.fps > 0
                and abs(meta.fps - S3_HZ) > 0.5):
            return max(1, int(round(n * S3_HZ / meta.fps)))
        return n
    dur = meta.duration
    if dur <= 0:
        return 1
    return max(1, int(round(dur * S3_HZ)))


# --------------------------------------------------------------------------------------
# Output sanitizers. Anything that reaches the scorer goes through these.
# --------------------------------------------------------------------------------------
def _as_int(x, default: int = 0) -> int:
    try:
        import numpy as np

        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return int(round(v))
    except Exception:
        return default


def sanitize_stage1(rows: Iterable[dict], ids: Sequence[str]):
    """One row per ID, always a legal label.

    IDs nothing was computed for get ``S1_LABELS[stable_bit(id)]`` rather than a fixed
    ORIGINAL: Stage 1 is scored MacroF1(ORIGINAL, RERECORDED) over *both* categories, so
    filling the gap with one constant caps that slice of the frame at 0.333 while an
    arbitrary balanced split reaches about 0.5 -- and the task is two-class by construction
    (the organizers generate the RERECORDED half themselves; the public example set is 5+5).
    See ``stable_bit`` for why the split is a CRC of the ID and not ``random``.
    """
    import pandas as pd

    got = {}
    for r in rows or []:
        i = str(r.get("ID", ""))
        a = str(r.get("answer", "")).upper().strip()
        if a in S1_LABELS:
            got[i] = a
    out = [{"ID": str(i), "answer": got.get(str(i), S1_LABELS[stable_bit(str(i))])}
           for i in ids]
    return pd.DataFrame(out, columns=["ID", "answer"])


def stable_bit(key: str) -> int:
    """A deterministic, platform-stable 0/1 from an ID (CRC32, not ``hash()``).

    Used as the *last-resort* value of the two categorical Stage 2 fields. Macro-F1 counts
    both categories, so a constant answer is the worst possible guess even when it is the
    majority one: against a 50/50 truth a constant scores 0.5*(2*0.5/1.5) + 0.5*0 = 0.333,
    while an arbitrary balanced split scores 0.5*0.5 + 0.5*0.5 = 0.5. The measured priors
    really are balanced -- 1048 AI Hub 597 clips give entry_side LEFT 50.3% / RIGHT 49.7% --
    so splitting on the ID is strictly better than picking a favourite, and being a pure
    function of the ID it stays reproducible across re-runs and machines.
    """
    import zlib

    return int(zlib.crc32(str(key).encode("utf-8")) & 1)


def s2_defaults(vid: str, frame_numbers: Sequence[int] | None, n_frames: int = 0) -> dict:
    """The Stage 2 answer for a clip nothing could be computed from. Never raises."""
    fn = list(frame_numbers or [])
    if fn:
        lo, hi, n = min(fn), max(fn), len(fn)
    else:
        n = max(1, int(n_frames or 1))
        lo, hi = 0, n - 1
    fps = s2_assumed_fps(n)
    c = int(round(lo + S2_COLLISION_REL * (hi - lo)))
    if fn:                                   # snap onto a frame number that really exists
        c = min(fn, key=lambda x: abs(x - c))
    e = max(lo, c - int(round(S2_ENTRY_LEAD_SEC * fps)))
    if fn:
        e = min((x for x in fn if x <= c), key=lambda x: abs(x - e), default=c)
    bit = stable_bit(vid)
    return {"collision_frame": int(c), "entry_frame": int(e),
            "evasion_space": int(bit), "entry_side": SIDE_LABELS[stable_bit(vid + "|side")]}


def sanitize_stage2(rows: Iterable[dict], ids: Sequence[str], n_frames: dict[str, int],
                    frame_numbers: dict[str, list[int]] | None = None):
    """Clamp frames into the video's own frame-number range and force valid categories.

    ``n_frames`` maps ID -> number of frames; ``frame_numbers`` (optional) maps ID -> the
    actual frame numbers parsed from filenames, which is what must be submitted when the
    folder does not start at 0 or skips numbers.
    """
    import pandas as pd

    got = {str(r.get("ID", "")): r for r in (rows or [])}
    out = []
    for i in ids:
        i = str(i)
        r = got.get(i, {})
        fn = (frame_numbers or {}).get(i)
        d = s2_defaults(i, fn, n_frames.get(i, 0))
        lo, hi = (min(fn), max(fn)) if fn else (0, max(0, int(n_frames.get(i, 1)) - 1))
        c = _as_int(r.get("collision_frame"), d["collision_frame"])
        c = max(lo, min(hi, c))
        # Default to the already-snapped s2_defaults entry rather than recomputing c - lead:
        # s2_defaults snaps onto a frame number that really exists, and on a folder that does
        # not start at 0 or skips numbers (frame_010000, step 7, ...) the arithmetic lands
        # between two real frames and submits a number no file has.
        e_default = d["entry_frame"] if not r else max(
            lo, c - int(round(S2_ENTRY_LEAD_SEC * s2_assumed_fps(len(fn) if fn else
                                                                 n_frames.get(i, 1)))))
        e = _as_int(r.get("entry_frame"), e_default)
        e = max(lo, min(hi, e))
        if fn and e not in fn:                   # snap onto a frame number that really exists
            e = min((x for x in fn if x <= c), key=lambda x: abs(x - e), default=c)
        if e > c:  # entry must precede collision
            e = c
        ev = _as_int(r.get("evasion_space"), d["evasion_space"])
        ev = d["evasion_space"] if ev not in (0, 1) else ev
        side = str(r.get("entry_side", "")).upper().strip()
        if side not in SIDE_LABELS:
            side = d["entry_side"]
        out.append({"ID": i, "collision_frame": int(c), "entry_frame": int(e),
                    "evasion_space": int(ev), "entry_side": side})
    df = pd.DataFrame(out, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
    for c in ("collision_frame", "entry_frame", "evasion_space"):
        df[c] = df[c].astype("int64")
    return df


# Class mix of the 1.2 M-sample comma2k19 label set (data/stage3_labels/manifest.csv):
# ACCELERATING 12.8% / DECELERATING 12.3% / CONSTANT 64.7% / STOPPED 10.1%,
# LEFT 7.3% / STRAIGHT 83.7% / RIGHT 9.0%.
S3_ACCEL_PRIOR = (("ACCELERATING", 0.128), ("DECELERATING", 0.123), ("STOPPED", 0.101),
                  ("CONSTANT", 0.648))
S3_STEER_PRIOR = (("LEFT", 0.073), ("RIGHT", 0.090), ("STRAIGHT", 0.837))


def s3_default_labels(n: int) -> tuple[list[str], list[str]]:
    """Filler labels for samples nothing was computed for.

    Not one constant class. Stage 3 is macro-F1 over all four accel and all three steer
    categories whether or not the submission uses them, so a class that is never predicted
    scores F1 = 0 and still divides the mean: filling a skipped video with 600 rows of
    CONSTANT/STRAIGHT both wastes those rows and inflates the dominant class's false
    positives. Laying the training prior down in contiguous blocks keeps every class alive
    and stays temporally plausible, at no cost to the videos that were predicted properly.
    """
    n = max(0, int(n))
    accel = [S3_ACCEL_PRIOR[-1][0]] * n
    steer = [S3_STEER_PRIOR[-1][0]] * n
    for arr, prior in ((accel, S3_ACCEL_PRIOR), (steer, S3_STEER_PRIOR)):
        start = 0
        for label, share in prior[:-1]:
            stop = min(n, start + int(round(share * n)))
            for k in range(start, stop):
                arr[k] = label
            start = stop
    return accel, steer


def sanitize_stage3(rows: Iterable[dict], counts: dict[str, int]):
    """Guarantee exactly ``counts[ID]`` contiguous sample_index rows per video."""
    import pandas as pd

    per: dict[str, dict[int, tuple[str, str]]] = {}
    for r in rows or []:
        i = str(r.get("ID", ""))
        k = _as_int(r.get("sample_index"), -1)
        if k < 0:
            continue
        a = str(r.get("accel_label", "")).upper().strip()
        s = str(r.get("steer_label", "")).upper().strip()
        a = a if a in ACCEL_LABELS else "CONSTANT"
        s = s if s in STEER_LABELS else "STRAIGHT"
        per.setdefault(i, {})[k] = (a, s)
    out = []
    for i, n in counts.items():
        i = str(i)
        have = per.get(i, {})
        n = max(1, int(n))
        # Lay the prior across the *gap*, not across the whole clip. s3_default_labels emits
        # the classes in contiguous blocks starting at 0, so indexing it by the absolute
        # sample number hands a partially-predicted video only the tail of that layout --
        # which is entirely the last (majority) class. A video the pacer abandoned half way
        # would then get 300 real predictions followed by 300 rows of pure CONSTANT/STRAIGHT,
        # losing exactly the every-class-stays-alive property this filler exists for.
        missing = [k for k in range(n) if k not in have]
        da, ds = s3_default_labels(len(missing))
        fill = {k: (da[j], ds[j]) for j, k in enumerate(missing)}
        for k in range(n):
            a, s = have.get(k) or fill.get(k, ("CONSTANT", "STRAIGHT"))
            out.append({"ID": i, "sample_index": int(k), "accel_label": a, "steer_label": s})
    df = pd.DataFrame(out, columns=["ID", "sample_index", "accel_label", "steer_label"])
    df["sample_index"] = df["sample_index"].astype("int64")
    return df


# --------------------------------------------------------------------------------------
# Content hashing -- duplicate media files must not straddle a train/val split.
# --------------------------------------------------------------------------------------
_HASH_EDGE = 1 << 20      # bytes read from each end in the cheap first pass


def _cheap_key(path: str) -> tuple | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(_HASH_EDGE))
            if size > 2 * _HASH_EDGE:
                fh.seek(-_HASH_EDGE, os.SEEK_END)
                h.update(fh.read(_HASH_EDGE))
    except OSError:
        return None
    return (size, h.hexdigest())


def _full_hash(path: str) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def content_hashes(paths: Iterable[str], workers: int = 8) -> dict[str, str]:
    """``path -> content id``, cheap for uniques and exact for duplicates.

    Pass 1 keys every file by ``(size, blake2b(first 1 MiB + last 1 MiB))``.  A key
    seen once is already unique -- two different videos cannot share a size AND both
    ends -- so it is returned as-is.  Only the files whose key repeats are read in
    full in pass 2, which is what makes "are these two mp4s byte-identical?"
    answerable over a 15k-file corpus without reading 80 GB.
    """
    paths = [str(p) for p in paths if p]
    uniq = sorted(set(paths))
    if not uniq:
        return {}

    def _pool(fn, items):
        if workers and workers > 1 and len(items) > 8:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(workers) as pool:
                return list(pool.imap(fn, items, chunksize=32))
        return [fn(i) for i in items]

    keys = _pool(_cheap_key, uniq)
    by_key: dict[tuple, list[str]] = {}
    for pth, k in zip(uniq, keys):
        if k is not None:
            by_key.setdefault(k, []).append(pth)
    out: dict[str, str] = {}
    dupes: list[str] = []
    for k, group in by_key.items():
        if len(group) == 1:
            out[group[0]] = f"c{k[0]:x}-{k[1]}"
        else:
            dupes.extend(group)
    if dupes:
        for pth, h in zip(dupes, _pool(_full_hash, dupes)):
            out[pth] = f"f{h}" if h else ""
    return out


# --------------------------------------------------------------------------------------
# Metrics that mirror the official scorer, for local validation.
# --------------------------------------------------------------------------------------
def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> float:
    from sklearn.metrics import f1_score

    if len(y_true) == 0:
        return 0.0
    labs = list(labels) if labels is not None else sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    return float(f1_score(list(map(str, y_true)), list(map(str, y_pred)), labels=labs,
                          average="macro", zero_division=0))


def acc_within(pred_sec: Sequence[float], true_sec: Sequence[float], tol: float = 0.3) -> float:
    """Stage 2 timing accuracy: |pred - true| <= tol seconds counts as correct."""
    import numpy as np

    if len(true_sec) == 0:
        return 0.0
    p = np.asarray(pred_sec, dtype=float)
    t = np.asarray(true_sec, dtype=float)
    ok = np.isfinite(p) & (np.abs(p - t) <= tol + 1e-9)
    return float(ok.mean())


def score_stage2(pred_c_sec, true_c_sec, pred_e_sec, true_e_sec,
                 pred_side, true_side, pred_ev, true_ev) -> dict:
    s_c = acc_within(pred_c_sec, true_c_sec)
    s_e = acc_within(pred_e_sec, true_e_sec)
    s_d = macro_f1(true_side, pred_side, SIDE_LABELS)
    s_v = macro_f1([str(int(x)) for x in true_ev], [str(int(x)) for x in pred_ev], ["0", "1"])
    return {"collision": s_c, "entry": s_e, "side": s_d, "evasion": s_v,
            "total": 0.35 * s_c + 0.35 * s_e + 0.15 * s_d + 0.15 * s_v}


def score_stage3(true_accel, pred_accel, true_steer, pred_steer) -> dict:
    """0.7 * macroF1(accel) + 0.3 * macroF1(steer over non-STOPPED ground truth)."""
    f_a = macro_f1(true_accel, pred_accel, ACCEL_LABELS)
    mask = [i for i, t in enumerate(true_accel) if str(t) != "STOPPED"]
    ts = [true_steer[i] for i in mask]
    ps = [pred_steer[i] for i in mask]
    f_s = macro_f1(ts, ps, STEER_LABELS) if ts else 0.0
    return {"accel": f_a, "steer": f_s, "total": 0.7 * f_a + 0.3 * f_s}


__all__ = [
    "set_offline_env", "seed_everything", "Budget", "PerItemPacer",
    "S1_LABELS", "ACCEL_LABELS", "STEER_LABELS", "SIDE_LABELS",
    "ACCEL_TO_IDX", "STEER_TO_IDX", "VIDEO_EXT", "IMAGE_EXT",
    "S3_HZ", "S3_FALLBACK_FPS", "GENERIC_FALLBACK_FPS",
    "S2_COLLISION_REL", "S2_ENTRY_LEAD_SEC", "s2_assumed_fps", "s2_defaults", "stable_bit",
    "resolve_stage_dir",
    "list_videos", "list_frame_dirs", "list_frames", "frame_number",
    "VideoMeta", "probe_video", "count_video_frames", "decode_frames", "s3_sample_count",
    "sanitize_stage1", "sanitize_stage2", "sanitize_stage3",
    "content_hashes",
    "macro_f1", "acc_within", "score_stage2", "score_stage3",
]
