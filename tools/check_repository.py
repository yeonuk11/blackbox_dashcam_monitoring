#!/usr/bin/env python3
"""Static checks for team boundaries, paths, and GitHub size safety."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 95 * 1024**2
REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}


def tracked_files() -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / value.decode() for value in process.stdout.split(b"\0") if value]


def main() -> int:
    errors: list[str] = []
    required = [
        ROOT / "submission/inference.py",
        ROOT / "submission/requirements.txt",
        ROOT / "tools/build_submission.py",
        ROOT / "tools/validate_submission.py",
    ]
    required.extend(
        ROOT / f"submission/model/stage{number}/predict.py" for number in (1, 2, 3)
    )
    required.extend(
        ROOT / f"submission/model/stage{number}/stage.json" for number in (1, 2, 3)
    )
    errors.extend(f"missing required file: {path.relative_to(ROOT)}" for path in required if not path.is_file())
    if (ROOT / "submission/script.py").exists():
        errors.append("submission/script.py must be removed; DACON supplies it")

    inference = ROOT / "submission/inference.py"
    if inference.is_file():
        tree = ast.parse(inference.read_text(encoding="utf-8"), filename=str(inference))
        names = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in sorted(REQUIRED_FUNCTIONS - names):
            errors.append(f"submission/inference.py missing {name}")

    absolute_home = re.compile(r"/(?:home|Users)/[^/\s'\"]+")
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if absolute_home.search(line):
                errors.append(
                    f"hard-coded user path: {path.relative_to(ROOT)}:{line_number}"
                )

    try:
        tracked = tracked_files()
    except subprocess.CalledProcessError as exc:
        errors.append(f"git ls-files failed: {exc}")
        tracked = []
    for path in tracked:
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(
                f"tracked file exceeds 95 MiB safety limit: {path.relative_to(ROOT)} "
                f"({path.stat().st_size / 1024**2:.1f} MiB)"
            )

    if errors:
        print("FAIL")
        for error in errors:
            print(f" - {error}")
        return 1
    print("PASS: repository skeleton")
    print(f" - tracked files checked: {len(tracked)}")
    print(" - no hard-coded user paths or tracked file over 95 MiB")
    print(" - all three stage interfaces and submission contract are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
