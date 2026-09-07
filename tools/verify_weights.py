#!/usr/bin/env python3
"""Verify local stage artifacts against tracked ``stage.json`` manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(stage_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = stage_dir / "stage.json"
    if not path.is_file():
        return None, [f"missing stage manifest: {path.relative_to(ROOT)}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid {path.relative_to(ROOT)}: {exc}"]
    if not isinstance(manifest, dict):
        return None, [f"stage manifest must be a JSON object: {path.relative_to(ROOT)}"]
    return manifest, []


def verify(*, allow_incomplete: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    verified: list[str] = []
    for stage in (1, 2, 3):
        stage_dir = SOURCE / "model" / f"stage{stage}"
        manifest, manifest_errors = load_manifest(stage_dir)
        errors.extend(manifest_errors)
        if manifest is None:
            continue
        if manifest.get("stage") != stage:
            errors.append(f"stage{stage}/stage.json has the wrong stage number")
        implemented = manifest.get("implemented") is True
        if not implemented and not allow_incomplete:
            errors.append(f"stage{stage} is not marked implemented")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"stage{stage}/stage.json must declare at least one artifact")
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append(f"stage{stage} has a non-object artifact entry")
                continue
            filename = artifact.get("filename")
            if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                errors.append(f"stage{stage} has an unsafe artifact filename: {filename!r}")
                continue
            path = stage_dir / filename
            required = artifact.get("required", True) is not False
            if not path.is_file():
                if required and not allow_incomplete:
                    errors.append(f"missing required artifact: {path.relative_to(ROOT)}")
                continue
            expected_bytes = artifact.get("bytes")
            expected_sha = artifact.get("sha256")
            if expected_bytes is None or expected_sha is None:
                if not allow_incomplete:
                    errors.append(f"stage{stage} artifact metadata is incomplete: {filename}")
                continue
            if path.stat().st_size != expected_bytes:
                errors.append(
                    f"size mismatch for {path.relative_to(ROOT)}: "
                    f"expected {expected_bytes}, got {path.stat().st_size}"
                )
                continue
            actual_sha = sha256(path)
            if actual_sha.lower() != str(expected_sha).lower():
                errors.append(f"SHA-256 mismatch for {path.relative_to(ROOT)}")
                continue
            verified.append(f"stage{stage}/{filename}")
    return errors, verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    errors, verified = verify(allow_incomplete=args.allow_incomplete)
    if errors:
        print("FAIL")
        for error in errors:
            print(f" - {error}")
        return 1
    print("PASS: stage manifests and local artifacts")
    for name in verified:
        print(f" - verified: {name}")
    if args.allow_incomplete:
        print(" - incomplete stages/artifacts were allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
