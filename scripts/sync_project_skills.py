#!/usr/bin/env python3
"""Validate and sync project-owned Codex skills into the runtime skill dir."""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_SKILLS = REPO_ROOT / "skills"
DEFAULT_RUNTIME_SKILLS = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "skills"
DEFAULT_VALIDATOR = DEFAULT_RUNTIME_SKILLS / ".system" / "skill-creator" / "scripts" / "quick_validate.py"


def _skill_dirs(root: Path, selected: list[str] | None = None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"skills directory not found: {root}")
    selected_set = set(selected or [])
    skills = sorted(path for path in root.iterdir() if (path / "SKILL.md").is_file())
    if selected_set:
        skills = [path for path in skills if path.name in selected_set]
        missing = sorted(selected_set - {path.name for path in skills})
        if missing:
            raise FileNotFoundError(f"selected skill(s) not found under {root}: {', '.join(missing)}")
    return skills


def _validate(skills: list[Path], validator: Path) -> None:
    if not validator.is_file():
        raise FileNotFoundError(f"skill validator not found: {validator}")
    for skill in skills:
        subprocess.run([sys.executable, str(validator), str(skill)], check=True)


def _compare_dirs(left: Path, right: Path) -> list[str]:
    if not right.exists():
        return [f"missing runtime skill: {right}"]
    if not right.is_dir():
        return [f"runtime path is not a directory: {right}"]

    diff = filecmp.dircmp(left, right)
    problems: list[str] = []
    for name in diff.left_only:
        problems.append(f"missing in runtime: {right / name}")
    for name in diff.right_only:
        problems.append(f"extra in runtime: {right / name}")
    for name in diff.diff_files:
        problems.append(f"file differs: {left / name} != {right / name}")
    for name in diff.funny_files:
        problems.append(f"cannot compare: {left / name} != {right / name}")
    for name, subdiff in diff.subdirs.items():
        problems.extend(_compare_dirs(left / name, right / name))
    return problems


def _check(skills: list[Path], runtime_root: Path) -> None:
    problems: list[str] = []
    for skill in skills:
        problems.extend(_compare_dirs(skill, runtime_root / skill.name))
    if problems:
        print("Project skill/runtime mismatch:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Project skills match runtime: {runtime_root}")


def _sync(skills: list[Path], runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        target = runtime_root / skill.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(skill, target)
        print(f"synced {skill.name} -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "sync", "validate"), nargs="?", default="check")
    parser.add_argument("--project-skills", type=Path, default=DEFAULT_PROJECT_SKILLS)
    parser.add_argument("--runtime-skills", type=Path, default=DEFAULT_RUNTIME_SKILLS)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--skill", action="append", help="Limit action to one skill name; may be repeated.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation before sync.")
    args = parser.parse_args()

    skills = _skill_dirs(args.project_skills, args.skill)
    if args.action == "validate":
        _validate(skills, args.validator)
        return 0
    if args.action == "sync":
        if not args.no_validate:
            _validate(skills, args.validator)
        _sync(skills, args.runtime_skills)
        return 0
    _check(skills, args.runtime_skills)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
