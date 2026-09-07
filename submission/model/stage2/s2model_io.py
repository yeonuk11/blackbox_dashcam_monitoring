"""One checkpoint-loading contract, shared by all three stages.

Every stage used to carry its own ``_find_checkpoint``.  Three copies drifted apart and two
of them had the same defect: ``Path(model_dir).rglob('*.pt')`` followed by *newest by mtime*.
That is a cross-stage hazard.  The evaluation server calls each stage with its own directory,
but nothing in the contract forbids handing all three the ``model/`` parent -- and our own
rehearsal harness does exactly that.  With ``rglob`` a Stage 1 run then walks into
``model/stage3/`` and picks whichever file a background trainer last wrote, which loads (or
worse, half-loads) the wrong network for the stage.

The rules here are deliberately narrow:

* **Only two directories are ever searched**: ``model_dir`` itself and ``model_dir/stageN``
  for the *requested* N.  No recursion, so a sibling stage's directory is unreachable.
* **A foreign stage directory is refused outright.**  Asked for stage 1 with ``model_dir``
  pointing at ``.../stage3``, the answer is None, not "the newest .pt in there".
* **Pretrained front-ends are not checkpoints.**  ``resnet18-f37072fd.pth`` sits in the
  organizers' own ``baseline/model/stage2/``, and RAFT rides along in ``model/stage3/``.
  Both match the torchvision release-file pattern ``<name>-<8 hex>.pth``, which no training
  script of ours emits.
* **Known-bad training artefacts are skipped by name** (``*smoke*``, ``*leaky*``, ...) so a
  deleted ``best.pt`` cannot silently promote the ablation that was left next to it.

Reading is defensive because the checkpoints are rewritten by trainers running *right now*:
``torch.save`` is not atomic, so a read can raise or -- far worse -- succeed with a truncated
state dict.  ``read_checkpoint`` retries and requires the file size to be unchanged across
the read; ``state_dict_report`` then compares the payload against the freshly-built module
before a single weight is trusted, and reports what differs instead of raising a bare
``RuntimeError`` from ``load_state_dict``.

Nothing in this module raises: every entry point answers ``None``/``(False, reason)`` and
prints one line to stderr, because the caller's alternative is always a heuristic that still
produces a legal submission.  A stage that dies takes the whole 60-minute run with it.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

__all__ = ["find_checkpoint", "read_checkpoint", "state_dict_report", "load_into",
           "call_filtered", "warn", "CKPT_SUFFIXES"]

CKPT_SUFFIXES = (".pt", ".pth")

# Preferred file names per stage, in the order they are tried. ``best.pt`` first: every
# trainer in scripts/ writes that name, and it is the one the eval logs quote.
_PREFERRED = {
    1: ("best.pt", "stage1.pt", "model.pt"),
    2: ("best.pt", "stage2.pt", "model.pt"),
    3: ("best.pt", "stage3.pt", "model.pt"),
}

# torchvision / timm release files: "<arch>-<8 hex>.pth". Our trainers never emit this.
_PRETRAINED_RE = re.compile(r"-[0-9a-f]{8}\.(pt|pth)$", re.IGNORECASE)
# Training artefacts that must never be promoted to "the" checkpoint by an mtime tiebreak.
_ARTEFACT_HINTS = ("smoke", "leaky", "debug", "scratch", "_tmp", "ablation", "optimizer")
_STAGE_DIR_RE = re.compile(r"^stage[_-]?([123])$", re.IGNORECASE)


def warn(who: str, msg: str) -> None:
    print(f"[{who}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def _is_candidate(p: Path) -> bool:
    if not p.is_file() or p.suffix.lower() not in CKPT_SUFFIXES:
        return False
    name = p.name.lower()
    if _PRETRAINED_RE.search(name):
        return False
    return not any(h in name for h in _ARTEFACT_HINTS)


def _stage_of_dir(d: Path) -> int | None:
    m = _STAGE_DIR_RE.match(d.name)
    return int(m.group(1)) if m else None


def find_checkpoint(model_dir, stage: int, who: str = "model_io") -> Path | None:
    """The one checkpoint stage ``stage`` may load from ``model_dir``, or None.

    Search order: the preferred names directly in ``model_dir``; the preferred names in
    ``model_dir/stageN``; then the newest plain ``*.pt``/``*.pth`` that is a *direct child*
    of either.  Never recursive, never another stage's directory.
    """
    stage = int(stage)
    d = Path(model_dir)

    if d.is_file():
        if not _is_candidate(d):
            warn(who, f"{d} is not a usable stage-{stage} checkpoint (pretrained/artefact "
                      f"name) -> no checkpoint")
            return None
        owner = _stage_of_dir(d.parent)
        if owner is not None and owner != stage:
            warn(who, f"refusing {d}: it lives in a stage-{owner} directory, not stage-{stage}")
            return None
        return d

    if not d.is_dir():
        warn(who, f"model_dir {d} does not exist -> no checkpoint")
        return None

    owner = _stage_of_dir(d)
    if owner is not None and owner != stage:
        warn(who, f"refusing to load stage-{stage} weights from a stage-{owner} directory "
                  f"({d}) -> no checkpoint")
        return None

    roots = [d]
    sub = d / f"stage{stage}"
    if sub.is_dir():
        roots.append(sub)

    for root in roots:
        for name in _PREFERRED[stage]:
            p = root / name
            if _is_candidate(p):
                return p

    loose: list[Path] = []
    for root in roots:
        try:
            loose += [p for p in root.iterdir() if _is_candidate(p)]
        except OSError:
            continue
    if not loose:
        warn(who, f"no stage-{stage} checkpoint under {d} "
                  f"(looked for {'/'.join(_PREFERRED[stage])} then loose *.pt)")
        return None
    pick = max(loose, key=lambda p: p.stat().st_mtime)
    if len(loose) > 1:
        warn(who, f"{len(loose)} loose checkpoints in {d}; taking the newest ({pick.name})")
    return pick


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def read_checkpoint(path, attempts: int = 3, pause: float = 1.0, who: str = "model_io"):
    """``torch.load`` a file another process may be rewriting. None if it never settles."""
    p = Path(path)
    last = None
    for i in range(max(1, int(attempts))):
        try:
            import torch

            size_before = p.stat().st_size
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            if p.stat().st_size != size_before:
                raise RuntimeError("file changed size while loading")
            if not isinstance(ck, dict):
                raise TypeError(f"expected a dict bundle, got {type(ck).__name__}")
            return ck
        except Exception as exc:                       # noqa: BLE001 - report, never raise
            last = exc
            warn(who, f"{p.name} unreadable on attempt {i + 1}/{attempts}: "
                      f"{type(exc).__name__}: {exc}")
            if i + 1 < attempts:
                time.sleep(pause)
    warn(who, f"giving up on {p}: {type(last).__name__ if last else 'unknown'}")
    return None


def _state_of(ck: dict):
    """The tensor dict inside a bundle, under whichever key the trainer chose."""
    for key in ("state", "state_dict", "model", "model_state_dict", "net", "weights"):
        v = ck.get(key)
        if isinstance(v, dict) and v:
            return v, key
    # A bare state dict saved with no wrapper: values are tensors, keys are strings.
    if ck and all(isinstance(k, str) for k in ck) and any(hasattr(v, "shape") for v in ck.values()):
        return ck, "<bare>"
    return None, None


def state_dict_report(net, state) -> tuple[bool, str]:
    """(usable, human-readable reason) for loading ``state`` into ``net``.

    Checked before ``load_state_dict`` so the failure names the mismatch.  A truncated
    pickle is the case that matters: it produces a *subset* of the keys, which
    ``strict=False`` would happily accept and then run a half-initialised network.
    """
    try:
        want = net.state_dict()
    except Exception as exc:                            # noqa: BLE001
        return False, f"module has no state_dict ({type(exc).__name__})"
    if not isinstance(state, dict) or not state:
        return False, "checkpoint carries no state dict"
    # Tolerate a DataParallel prefix, which is a wrapper artefact and not a mismatch.
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing = [k for k in want if k not in state]
    unexpected = [k for k in state if k not in want]
    bad_shape = [k for k in want
                 if k in state and tuple(getattr(state[k], "shape", ())) != tuple(want[k].shape)]
    if missing or unexpected or bad_shape:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing (e.g. {missing[0]})")
        if unexpected:
            parts.append(f"{len(unexpected)} unexpected (e.g. {unexpected[0]})")
        if bad_shape:
            k = bad_shape[0]
            parts.append(f"{len(bad_shape)} shape mismatches (e.g. {k}: checkpoint "
                         f"{tuple(state[k].shape)} vs model {tuple(want[k].shape)})")
        return False, "; ".join(parts)
    return True, f"{len(want)} tensors match"


def load_into(net, ck: dict, who: str = "model_io") -> tuple[bool, str]:
    """Validate then ``load_state_dict(strict=True)``. Never raises."""
    state, key = _state_of(ck if isinstance(ck, dict) else {})
    if state is None:
        return False, "no recognised state-dict key in the bundle"
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    ok, reason = state_dict_report(net, state)
    if not ok:
        return False, f"state['{key}'] does not match the model class: {reason}"
    try:
        net.load_state_dict(state, strict=True)
    except Exception as exc:                            # noqa: BLE001
        return False, f"load_state_dict failed after validation: {type(exc).__name__}: {exc}"
    return True, reason


# --------------------------------------------------------------------------------------
def call_filtered(fn, *pos, **kw):
    """Call ``fn`` with only the keyword arguments its signature accepts.

    The stage modules are written by different pipelines and their loaders do not agree on
    whether they take ``budget``/``deadline``; passing an unexpected keyword would drop the
    whole stage to defaults over a signature detail.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*pos, **kw)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*pos, **kw)
    return fn(*pos, **{k: v for k, v in kw.items() if k in params})
