"""Stage 1 placeholder; replace on a ``stage1/*`` branch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
COLUMNS = ["ID", "answer"]


def _videos(data_dir: Path) -> list[Path]:
    root = data_dir / "videos" if (data_dir / "videos").is_dir() else data_dir
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def predict(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Return a structurally valid placeholder prediction.

    ``model_dir`` is part of the stable team API. The placeholder intentionally
    does not load a checkpoint and must be replaced before a scored submission.
    """
    del model_dir
    rows = [{"ID": path.stem, "answer": "ORIGINAL"} for path in _videos(data_dir)]
    return pd.DataFrame(rows, columns=COLUMNS)
