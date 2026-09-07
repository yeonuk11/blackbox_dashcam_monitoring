"""Stage 2 placeholder; replace on a ``stage2/*`` branch."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Return one structurally valid row per Stage 2 sample directory."""
    del model_dir
    root = data_dir / "images" if (data_dir / "images").is_dir() else data_dir
    rows = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        frames = sorted(
            (path for path in folder.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=_frame_number,
        )
        if not frames:
            continue
        rows.append({
            "ID": folder.name,
            "collision_frame": _frame_number(frames[-1]),
            "entry_frame": _frame_number(frames[0]),
            "evasion_space": 0,
            "entry_side": "LEFT",
        })
    return pd.DataFrame(rows, columns=COLUMNS)
