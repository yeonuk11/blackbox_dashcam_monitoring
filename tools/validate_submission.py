#!/usr/bin/env python3
"""Validate the DACON three-stage submission archive contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_ROOT = {"inference.py", "requirements.txt"}
REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}
MAX_ZIP_BYTES = 10 * 1024**3
MAX_UNPACKED_BYTES = 32 * 1024**3


def _validate_stage_manifest(
    archive: zipfile.ZipFile, names: set[str], stage: int, *, require_weights: bool
) -> list[str]:
    errors: list[str] = []
    manifest_name = f"model/stage{stage}/stage.json"
    if manifest_name not in names:
        return [f"missing stage manifest: {manifest_name}"]
    try:
        manifest = json.loads(archive.read(manifest_name))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [f"invalid {manifest_name}: {exc}"]
    if not isinstance(manifest, dict) or manifest.get("stage") != stage:
        errors.append(f"{manifest_name} has an invalid stage declaration")
        return errors
    if require_weights and manifest.get("implemented") is not True:
        errors.append(f"stage{stage} is not marked implemented")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{manifest_name} must declare at least one artifact")
        return errors
    filenames: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append(f"stage{stage} has a non-object artifact entry")
            continue
        filename = artifact.get("filename")
        if not isinstance(filename, str) or not filename or PurePosixPath(filename).name != filename:
            errors.append(f"stage{stage} has an unsafe artifact filename: {filename!r}")
            continue
        filenames.add(filename)
        member_name = f"model/stage{stage}/{filename}"
        required = artifact.get("required", True) is not False
        if require_weights and required and member_name not in names:
            errors.append(f"missing required artifact: {member_name}")
            continue
        if require_weights and required:
            expected_sha = artifact.get("sha256")
            expected_bytes = artifact.get("bytes")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                errors.append(f"missing SHA-256 metadata for {member_name}")
                continue
            payload = archive.read(member_name)
            if expected_bytes != len(payload):
                errors.append(f"size mismatch for {member_name}")
            if hashlib.sha256(payload).hexdigest().lower() != expected_sha.lower():
                errors.append(f"SHA-256 mismatch for {member_name}")
    if require_weights and "best.pt" not in filenames:
        errors.append(f"stage{stage} manifest must declare best.pt")
    return errors


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
        for stage in (1, 2, 3):
            stage_prefix = f"model/stage{stage}/"
            if not any(name.startswith(stage_prefix) for name in names):
                errors.append(f"missing stage package: {stage_prefix}")
            errors.extend(
                _validate_stage_manifest(archive, names, stage, require_weights=require_weights)
            )
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
    print(" - stage manifests and top-level archive layout are valid")
    if args.require_weights:
        print(" - all stages are implemented and artifact hashes match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
