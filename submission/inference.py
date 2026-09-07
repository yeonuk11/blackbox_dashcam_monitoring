"""Stable DACON three-stage inference entry points.

Model implementation stays inside ``model/stageN``. This shared adapter only
dispatches calls and validates the team-wide output contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.stage1 import predict as _predict_stage1
from model.stage2 import predict as _predict_stage2
from model.stage3 import predict as _predict_stage3


STAGE_COLUMNS = {
    1: ["ID", "answer"],
    2: ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    3: ["ID", "sample_index", "accel_label", "steer_label"],
}
STAGE1_LABELS = {"ORIGINAL", "RERECORDED"}
STAGE3_ACCEL_LABELS = {"STOPPED", "ACCELERATING", "DECELERATING", "CONSTANT"}
STAGE3_STEER_LABELS = {"LEFT", "STRAIGHT", "RIGHT"}


def _validate_result(stage: int, result: pd.DataFrame) -> pd.DataFrame:
    """Fail early when a stage violates the shared DACON output contract."""
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Stage {stage} must return pandas.DataFrame, got {type(result)!r}")
    expected = STAGE_COLUMNS[stage]
    if list(result.columns) != expected:
        raise ValueError(
            f"Stage {stage} columns must be {expected}, got {list(result.columns)}"
        )
    if result["ID"].isna().any():
        raise ValueError(f"Stage {stage} output contains a missing ID")

    if stage in (1, 2) and result["ID"].duplicated().any():
        raise ValueError(f"Stage {stage} output contains duplicate IDs")
    if stage == 1:
        unknown = set(result["answer"].dropna().unique()) - STAGE1_LABELS
        if result["answer"].isna().any() or unknown:
            raise ValueError(f"Stage 1 contains invalid answer labels: {sorted(unknown)}")
    if stage == 3:
        if result[["ID", "sample_index"]].duplicated().any():
            raise ValueError("Stage 3 output contains duplicate (ID, sample_index) rows")
        unknown_accel = set(result["accel_label"].dropna().unique()) - STAGE3_ACCEL_LABELS
        unknown_steer = set(result["steer_label"].dropna().unique()) - STAGE3_STEER_LABELS
        if result[["sample_index", "accel_label", "steer_label"]].isna().any().any():
            raise ValueError("Stage 3 output contains a missing required value")
        if unknown_accel or unknown_steer:
            raise ValueError(
                "Stage 3 contains invalid labels: "
                f"accel={sorted(unknown_accel)}, steer={sorted(unknown_steer)}"
            )
    return result.reset_index(drop=True)


def predict_stage1(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _validate_result(1, _predict_stage1(Path(data_dir), Path(model_dir)))


def predict_stage2(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _validate_result(2, _predict_stage2(Path(data_dir), Path(model_dir)))


def predict_stage3(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    return _validate_result(3, _predict_stage3(Path(data_dir), Path(model_dir)))


__all__ = ["predict_stage1", "predict_stage2", "predict_stage3"]
