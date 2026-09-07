#!/usr/bin/env python3
"""Build a clean DACON submission ZIP from the ``submission`` directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from validate_submission import validate
from verify_weights import verify


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


def build(output: Path, *, allow_incomplete: bool) -> None:
    required = [SOURCE / "inference.py", SOURCE / "requirements.txt"]
    for number in (1, 2, 3):
        stage_dir = SOURCE / "model" / f"stage{number}"
        required.extend((stage_dir / "__init__.py", stage_dir / "stage.json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required source files: {missing}")

    artifact_errors, verified = verify(allow_incomplete=allow_incomplete)
    if artifact_errors:
        raise RuntimeError("stage artifact verification failed: " + "; ".join(artifact_errors))

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members():
            archive.write(path, path.relative_to(SOURCE).as_posix())
    errors = validate(output, require_weights=not allow_incomplete)
    if errors:
        output.unlink(missing_ok=True)
        raise RuntimeError("invalid submission: " + "; ".join(errors))
    print(f"built: {output.resolve()}")
    print(f"compressed_size_mib: {output.stat().st_size / 1024**2:.2f}")
    print(f"mode: {'skeleton' if allow_incomplete else 'final'}")
    for name in verified:
        print(f"verified: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "submission.zip")
    parser.add_argument(
        "--allow-incomplete", "--allow-missing-weights",
        dest="allow_incomplete", action="store_true",
        help="build a development skeleton even when stages or artifacts are incomplete",
    )
    args = parser.parse_args()
    build(args.output, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
