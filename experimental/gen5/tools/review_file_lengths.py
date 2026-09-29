#!/usr/bin/env python3
"""Prompt architectural review when a source file becomes unusually long."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ReviewLevel(IntEnum):
    NONE = 0
    CONSIDER_SPLIT = 1
    STRONGLY_CONSIDER_SPLIT = 2


@dataclass(frozen=True)
class Policy:
    review_at: int = 1_400
    strong_review_at: int = 2_000
    suffixes: frozenset[str] = frozenset(
        {
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".js",
            ".jsx",
            ".metal",
            ".m",
            ".mm",
            ".py",
            ".rs",
            ".swift",
            ".ts",
            ".tsx",
        }
    )
    ignored_directories: frozenset[str] = frozenset(
        {
            ".build",
            ".git",
            ".swiftpm",
            ".venv",
            "generated",
            "node_modules",
            "target",
            "vendor",
        }
    )


def iter_source_files(root: Path, policy: Policy):
    for directory, child_directories, file_names in os.walk(root):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name not in policy.ignored_directories
        )
        for file_name in sorted(file_names):
            path = Path(directory, file_name)
            if path.is_file() and not path.is_symlink():
                if path.suffix.lower() in policy.suffixes:
                    yield path


def count_lines(path: Path) -> int:
    data = path.read_bytes()
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def classify(lines: int, policy: Policy) -> ReviewLevel:
    if lines >= policy.strong_review_at:
        return ReviewLevel.STRONGLY_CONSIDER_SPLIT
    if lines >= policy.review_at:
        return ReviewLevel.CONSIDER_SPLIT
    return ReviewLevel.NONE


def review_repository(
    root: Path, policy: Policy
) -> tuple[int, list[tuple[Path, int, ReviewLevel]]]:
    reviewed: list[tuple[Path, int, ReviewLevel]] = []
    checked = 0
    for path in sorted(iter_source_files(root, policy)):
        checked += 1
        lines = count_lines(path)
        level = classify(lines, policy)
        if level is not ReviewLevel.NONE:
            reviewed.append((path.relative_to(root), lines, level))
    return checked, reviewed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    policy = Policy()
    checked, reviewed = review_repository(root, policy)

    print(f"source length review: {checked} files checked")
    for path, lines, level in reviewed:
        if level is ReviewLevel.STRONGLY_CONSIDER_SPLIT:
            prompt = "strongly consider a responsibility-based split"
        else:
            prompt = "consider whether responsibilities should be split"
        print(f"- {path}: {lines} lines; {prompt}")
    if not reviewed:
        print("- no source file currently needs a size review")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
