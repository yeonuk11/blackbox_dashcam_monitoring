#!/usr/bin/env python3
"""Build a clean DACON submission ZIP from the ``submission`` directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from validate_submission import validate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def members() -> list[Path]:
    return sorted(
        path for path in SOURCE.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name != "script.py"
    )


def build(output: Path, *, allow_missing_weights: bool) -> None:
    required = [SOURCE / "inference.py", SOURCE / "requirements.txt"]
    required.extend(SOURCE / "model" / f"stage{number}" / "__init__.py" for number in (1, 2, 3))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required source files: {missing}")
    weight_paths = [SOURCE / "model" / f"stage{number}" / "best.pt" for number in (1, 2, 3)]
    missing_weights = [str(path) for path in weight_paths if not path.is_file()]
    if missing_weights and not allow_missing_weights:
        raise FileNotFoundError(
            "final build requires each stage best.pt; use --allow-missing-weights "
            f"only for skeleton validation: {missing_weights}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members():
            archive.write(path, path.relative_to(SOURCE).as_posix())
    errors = validate(output, require_weights=not allow_missing_weights)
    if errors:
        output.unlink(missing_ok=True)
        raise RuntimeError("invalid submission: " + "; ".join(errors))
    print(f"built: {output.resolve()}")
    print(f"compressed_size_mib: {output.stat().st_size / 1024**2:.2f}")
    if missing_weights:
        print("mode: skeleton (weights intentionally absent)")
    else:
        print("mode: final (all stage weights present)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "submission.zip")
    parser.add_argument("--allow-missing-weights", action="store_true")
    args = parser.parse_args()
    build(args.output, allow_missing_weights=args.allow_missing_weights)


if __name__ == "__main__":
    main()
