"""Stage 3 placeholder; replace on a ``stage3/*`` branch."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
COLUMNS = ["ID", "sample_index", "accel_label", "steer_label"]


def _videos(data_dir: Path) -> list[Path]:
    root = data_dir / "videos" if (data_dir / "videos").is_dir() else data_dir
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def predict(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Return neutral motion labels while preserving the required row shape."""
    del model_dir
    rows = []
    for path in _videos(data_dir):
        capture = cv2.VideoCapture(str(path))
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        rows.extend(
            {
                "ID": path.stem,
                "sample_index": sample_index,
                "accel_label": "CONSTANT",
                "steer_label": "STRAIGHT",
            }
            for sample_index in range(frame_count)
        )
    return pd.DataFrame(rows, columns=COLUMNS)
