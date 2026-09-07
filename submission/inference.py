"""DACON three-stage inference entry points.

The evaluator imports these three public functions. Stage implementations are
isolated under ``model/stageN`` so each team member can work independently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.stage1 import predict as _predict_stage1
from model.stage2 import predict as _predict_stage2
from model.stage3 import predict as _predict_stage3


def predict_stage1(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _predict_stage1(Path(data_dir), Path(model_dir))


def predict_stage2(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _predict_stage2(Path(data_dir), Path(model_dir))


def predict_stage3(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _predict_stage3(Path(data_dir), Path(model_dir))


__all__ = ["predict_stage1", "predict_stage2", "predict_stage3"]
