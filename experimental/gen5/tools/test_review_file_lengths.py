"""Regression tests for source-file length review reminders."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.review_file_lengths import Policy, ReviewLevel, classify, review_repository


class FileLengthReviewTests(unittest.TestCase):
    def test_thresholds_are_advisory_bands(self) -> None:
        policy = Policy()
        cases = {
            1_399: ReviewLevel.NONE,
            1_400: ReviewLevel.CONSIDER_SPLIT,
            1_999: ReviewLevel.CONSIDER_SPLIT,
            2_000: ReviewLevel.STRONGLY_CONSIDER_SPLIT,
            20_000: ReviewLevel.STRONGLY_CONSIDER_SPLIT,
        }
        for lines, expected in cases.items():
            with self.subTest(lines=lines):
                self.assertEqual(classify(lines, policy), expected)

    def test_repository_review_lists_long_source_without_failing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "short.rs").write_text("line\n" * 20, encoding="utf-8")
            (root / "long.py").write_text("line\n" * 1_401, encoding="utf-8")
            (root / "target").mkdir()
            (root / "target" / "ignored.rs").write_text(
                "line\n" * 2_001, encoding="utf-8"
            )
            checked, reviewed = review_repository(root, Policy())

        self.assertEqual(checked, 2)
        self.assertEqual(
            reviewed,
            [(Path("long.py"), 1_401, ReviewLevel.CONSIDER_SPLIT)],
        )

    def test_repository_review_excludes_root_and_nested_virtual_environments(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "apple" / "study"
            project.mkdir(parents=True)
            (project / "export.py").write_text("line\n" * 1_401, encoding="utf-8")
            for parent in (root, project):
                packages = parent / ".venv" / "lib" / "python3.11" / "site-packages"
                packages.mkdir(parents=True)
                (packages / "dependency.py").write_text(
                    "line\n" * 2_001, encoding="utf-8"
                )
            checked, reviewed = review_repository(root, Policy())

        self.assertEqual(checked, 1)
        self.assertEqual(
            reviewed,
            [(Path("apple/study/export.py"), 1_401, ReviewLevel.CONSIDER_SPLIT)],
        )


if __name__ == "__main__":
    unittest.main()
