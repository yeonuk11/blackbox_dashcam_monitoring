#!/usr/bin/env python3
"""Validate the DACON three-stage submission archive contract."""

from __future__ import annotations

import argparse
import ast
import sys
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_ROOT = {"inference.py", "requirements.txt"}
REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}
STAGE_DIRS = {f"model/stage{number}/" for number in (1, 2, 3)}
MAX_ZIP_BYTES = 10 * 1024**3
MAX_UNPACKED_BYTES = 32 * 1024**3


def validate(path: Path, *, require_weights: bool = False) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"archive not found: {path}"]
    if path.stat().st_size > MAX_ZIP_BYTES:
        errors.append("compressed archive exceeds 10 GiB")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"invalid ZIP: {exc}"]
    with archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        if sum(info.file_size for info in infos) > MAX_UNPACKED_BYTES:
            errors.append("unpacked archive exceeds 32 GiB")
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                errors.append(f"unsafe archive path: {info.filename}")
        missing = sorted(REQUIRED_ROOT - names)
        if missing:
            errors.append(f"missing root files: {missing}")
        if "script.py" in names:
            errors.append("script.py must not be submitted; DACON supplies it")
        top = {PurePosixPath(name).parts[0] for name in names if name}
        unexpected = sorted(top - {"inference.py", "requirements.txt", "model"})
        if unexpected:
            errors.append(f"unexpected top-level entries: {unexpected}")
        for stage_dir in sorted(STAGE_DIRS):
            if not any(name.startswith(stage_dir) for name in names):
                errors.append(f"missing stage package: {stage_dir}")
            if require_weights and f"{stage_dir}best.pt" not in names:
                errors.append(f"missing final weight: {stage_dir}best.pt")
        if "inference.py" in names:
            try:
                tree = ast.parse(archive.read("inference.py"), filename="inference.py")
                functions = {
                    node.name: node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"invalid inference.py: {exc}")
            else:
                for name in sorted(REQUIRED_FUNCTIONS - functions.keys()):
                    errors.append(f"missing function: {name}")
                for name in sorted(REQUIRED_FUNCTIONS & functions.keys()):
                    positional = functions[name].args.posonlyargs + functions[name].args.args
                    if len(positional) < 2:
                        errors.append(f"{name} must accept data_dir and model_dir")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("dist/submission.zip"))
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()
    errors = validate(args.archive, require_weights=args.require_weights)
    if errors:
        print("FAIL")
        for error in errors:
            print(f" - {error}")
        return 1
    print(f"PASS: {args.archive}")
    print(" - root files and all three predict functions are present")
    print(" - model/stage1, stage2, stage3 packages are present")
    if args.require_weights:
        print(" - all three best.pt files are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
