"""Launcher contract tests, with bootstrap and exec mocked; no live/GPU process."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paisho_live_launch as launcher
from test_paisho_teacher_bootstrap import write_checkpoint


class LiveLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.campaign = self.root / "bootstrap"
        self.campaign.mkdir()
        (self.campaign / "plan.json").write_text("{}")
        self.selected = self.campaign / "selected.ckpt"
        write_checkpoint(self.selected, step=164, generation=49)
        self.executable = self.root / "paisho-live"
        self.executable.write_text("fake")
        self.executable.chmod(0o700)
        self.options = ["--campaign-dir", str(self.root / "live"), "--target-generation", "100",
                        "--workers", "3", "--durable-every", "5"]

    def result(self, passed):
        return {"status": "gate_passed" if passed else "budget_exhausted", "gate_passed": passed,
            "teacher_stopped_permanently": True, "selected_checkpoint": str(self.selected),
            "selected_checkpoint_sha256": launcher.bootstrap.digest(self.selected),
            "evaluation": {"conclusion": launcher.bootstrap.UPPER if passed else "inconclusive-maximum-eligible-pairs"}}

    def test_v5_requires_strict_live_selection(self):
        plan = {"champion_only": True, "minimum_teacher_cycles": 1,
                "initial_checkpoint": {"path": "parent"}}
        (self.campaign / "plan.json").write_text(json.dumps(plan))
        result = self.result(True)
        result["cycles_completed"] = 1
        with patch.object(launcher.bootstrap, "run", return_value=result), self.assertRaisesRegex(ValueError, "champion-only"):
            launcher.launch(self.campaign, self.executable, self.options)
        with patch.object(launcher.bootstrap, "run", return_value=result), patch.object(launcher.os, "execv") as execute, contextlib.redirect_stderr(io.StringIO()):
            launcher.launch(self.campaign, self.executable, self.options + ["--champion-only", "true"])
        self.assertEqual(execute.call_args.args[1][-1], "site")

    def test_v5_parent_upper_at_exhaustion_still_starts_random(self):
        plan = {"champion_only": True, "minimum_teacher_cycles": 1,
                "initial_checkpoint": {"path": str(self.selected)}}
        (self.campaign / "plan.json").write_text(json.dumps(plan))
        result = self.result(False)
        result.update({"cycles_completed": 5, "evaluation": {"conclusion": launcher.bootstrap.UPPER}})
        with patch.object(launcher.bootstrap, "run", return_value=result), patch.object(launcher.os, "execv") as execute, contextlib.redirect_stderr(io.StringIO()):
            launcher.launch(self.campaign, self.executable, self.options + ["--champion-only", "true"])
        self.assertEqual(execute.call_args.args[1][-1], "random")

    def test_passed_site_and_generation_from_selected_checkpoint(self):
        err = io.StringIO()
        with patch.object(launcher.bootstrap, "run", return_value=self.result(True)) as run, \
             patch.object(launcher.os, "execv") as execute, contextlib.redirect_stderr(err):
            launcher.launch(self.campaign, self.executable, self.options)
        run.assert_called_once_with(self.campaign)
        execute.assert_called_once_with(str(self.executable), [str(self.executable), *self.options,
            "--initial-checkpoint", str(self.selected), "--opponent", "site"])
        handoff = json.loads(err.getvalue())
        self.assertEqual(handoff["first_live_generation_if_new"], 50)
        self.assertEqual(handoff["target_generation"], 100)

    def test_exhausted_is_random_not_passed(self):
        with patch.object(launcher.bootstrap, "run", return_value=self.result(False)) as run, \
             patch.object(launcher.os, "execv") as execute, contextlib.redirect_stderr(io.StringIO()):
            launcher.launch(self.campaign, self.executable, self.options)
        self.assertEqual(run.call_count, 1)  # No endless teacher retries after cap.
        self.assertEqual(execute.call_args.args[1][-2:], ["--opponent", "random"])

    def test_overrides_rejected_before_bootstrap(self):
        for addition in (["--opponent", "site"], ["--initial-checkpoint", "x"],
                         ["--opponent=random", "x"], ["--initial-checkpoint=x", "x"]):
            with self.subTest(addition=addition), patch.object(launcher.bootstrap, "run") as run:
                with self.assertRaises(ValueError):
                    launcher.launch(self.campaign, self.executable, self.options + addition)
                run.assert_not_called()

    def test_missing_or_duplicate_required_options(self):
        for options in ([], self.options + ["--target-generation", "101"], ["--campaign-dir"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.forwarded_options(options)

    def test_no_directory_migration_or_nesting(self):
        options = self.options.copy()
        options[1] = str(self.campaign)
        with patch.object(launcher.bootstrap, "run") as run, self.assertRaisesRegex(ValueError, "separate"):
            launcher.launch(self.campaign, self.executable, options)
        run.assert_not_called()

    def test_failed_bootstrap_is_not_retried_or_execed(self):
        with patch.object(launcher.bootstrap, "run", side_effect=RuntimeError("child failed")) as run, \
             patch.object(launcher.os, "execv") as execute, contextlib.redirect_stderr(io.StringIO()):
            code = launcher.main(["--bootstrap-campaign", str(self.campaign), "--live-executable",
                                  str(self.executable), "--", *self.options])
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)
        execute.assert_not_called()

    def test_inconsistent_result_cannot_promote_tier(self):
        result = self.result(False)
        result["status"] = "gate_passed"
        with patch.object(launcher.bootstrap, "run", return_value=result), self.assertRaises(ValueError):
            launcher.launch(self.campaign, self.executable, self.options)

    def test_changed_checkpoint_rejected(self):
        result = self.result(True)
        write_checkpoint(self.selected, generation=50)
        with patch.object(launcher.bootstrap, "run", return_value=result), self.assertRaisesRegex(ValueError, "provenance"):
            launcher.launch(self.campaign, self.executable, self.options)

    def test_target_is_absolute_not_rebased(self):
        options = self.options.copy()
        options[3] = "48"
        with patch.object(launcher.bootstrap, "run", return_value=self.result(True)), self.assertRaisesRegex(ValueError, "precedes"):
            launcher.launch(self.campaign, self.executable, options)


if __name__ == "__main__":
    unittest.main()
