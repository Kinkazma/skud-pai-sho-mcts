import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import paisho_compact_history as history


def result(delta=250, budget=32, reference=32, kind="legacy-cpu-heuristic"):
    return {"identity": {"candidate_budget": budget, "reference_budget": reference, "reference_kind":kind},
            "failed": False, "complete_archive": True,
            "paired_results": {"eligible_pairs": 8, "scheduled_pairs": 8},
            "regularized_pair_delta": [{"prior_sd": 300, "posterior_mode": delta,
                                         "credible_95": [-100, 500]}]}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        training = self.root / "training"
        history.write(training / "progress.json", {"published_version": 7})
        history.write(training / "models/version-00000007.json", {"training_steps": 99, "weights": [1]})
        binary = self.root / "binary"
        binary.write_text("frozen executable")
        self.args = argparse.Namespace(training=training, output=self.root / "history", binary=binary,
            pairs=8, workers=2, seconds=600, decision_limit=2048, human_anchor=None, anchor_cohort=None)

    def test_sweep_freezes_published_version_and_finishes_pending_before_new_model(self):
        sweep, manifest = history.prepare_sweep(self.args)
        self.assertEqual(manifest["published_version"], 7)
        history.write(self.args.training / "models/version-00000008.json", {"weights": [2]})
        history.write(self.args.training / "progress.json", {"published_version": 8})
        other, again = history.prepare_sweep(self.args)
        self.assertEqual(other, sweep)
        self.assertEqual(again["model_sha256"], manifest["model_sha256"])
        history.write(sweep / "finished.json", {})
        newer, newest = history.prepare_sweep(self.args)
        self.assertNotEqual(newer, sweep)
        self.assertNotEqual(newest["model_sha256"], manifest["model_sha256"])
        self.assertEqual(newest["published_version"], 8)

    def test_same_model_is_not_retested_and_changed_protocol_is_rejected(self):
        sweep, _ = history.prepare_sweep(self.args)
        history.write(sweep / "finished.json", {})
        self.assertEqual(history.prepare_sweep(self.args)[0], sweep)
        self.args.seconds = 601
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            history.prepare_sweep(self.args)

    def test_job_list_uses_one_512_self_reference_for_both_roles(self):
        self.assertEqual(len(history.jobs()), 9)
        self.assertEqual(history.jobs().count((512, 512)), 1)
        self.assertEqual(history.jobs()[:5], [(b, b) for b in history.BUDGETS])

    def test_milestones_require_complete_full_pairs_same_budget(self):
        baseline = result()
        self.assertEqual(history.milestone_thresholds(baseline, set()), [100, 200])
        self.assertEqual(history.milestone_thresholds(baseline, {100}), [200])
        changes = [{"failed": True}, {"complete_archive": False},
                   {"paired_results": {"eligible_pairs": 7, "scheduled_pairs": 8}},
                   {"paired_results": {"eligible_pairs": 7, "scheduled_pairs": 7}},
                   {"identity": {"candidate_budget": 32, "reference_budget": 512}},
                   {"regularized_pair_delta": []}]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(history.milestone_thresholds({**baseline, **change}, set()), [])
        self.assertEqual(history.milestone_thresholds(result(-200), set()), [])
        self.assertEqual(history.milestone_thresholds(result(kind="compact-model"), set()), [])
        corrected=result()
        corrected["identity"]["rules"]=history.HISTORICAL_RULES+"-v2"
        self.assertEqual(history.milestone_thresholds(corrected,set()),[])

    def test_historical_marks_preserve_first_model_across_new_weights(self):
        sweep, _ = history.prepare_sweep(self.args)
        marks = history.record_milestones(self.args.output, sweep, {"runs": [result()]})
        self.assertEqual(len(marks), 2)
        first = Path(marks[0]) / "mark.json"
        old = first.read_bytes()
        snapshot_hash = history.digest(Path(marks[0]) / "model.json")
        history.write(sweep / "model.json", {"weights": [5]})
        self.assertEqual(history.record_milestones(self.args.output, sweep, {"runs": [result(320)]}),
                         [str(self.args.output / "milestones/mcts32/plus-0300")])
        self.assertEqual(first.read_bytes(), old)
        self.assertEqual(history.digest(Path(marks[0]) / "model.json"), snapshot_hash)
        self.assertFalse(history.read(first)["stops_training"])
        self.assertIsNone(history.read(first)["canonical_davidson_rating"])

    def test_partial_comparison_is_archived_without_automatic_retry(self):
        sweep, manifest = history.prepare_sweep(self.args)
        for job in manifest["jobs"]:
            (sweep / job["directory"]).mkdir()
        with patch.object(history, "update_sweep"), patch.object(history.subprocess, "run") as run:
            history.run_sweep(self.args)
            run.assert_not_called()
        receipt = history.read(sweep / (manifest["jobs"][0]["directory"] + ".receipt.json"))
        self.assertEqual(receipt["status"], "interrupted-no-retry")
        self.assertIsNone(receipt["returncode"])

    def test_native_jobs_bind_frozen_model_and_seeds_without_collection_cutoffs(self):
        child = Mock()
        child.wait.return_value = 0
        child.poll.return_value = 0
        with patch.object(history, "update_sweep"), patch.object(history.subprocess, "Popen", return_value=child) as spawn:
            sweep = history.run_sweep(self.args)
        self.assertEqual(spawn.call_count, 9)
        commands = [call.args[0] for call in spawn.call_args_list]
        self.assertEqual(len({command[command.index("--candidate") + 1] for command in commands}), 1)
        self.assertEqual(len({command[command.index("--first-pair") + 1] for command in commands}), 9)
        self.assertTrue(all("--game-seconds" not in command and "--move-ms" not in command for command in commands))
        self.assertTrue((sweep / "finished.json").exists())
        with patch.object(history, "update_sweep"), patch.object(history.subprocess, "Popen") as repeated:
            history.run_sweep(self.args)
            repeated.assert_not_called()

    def test_failed_training_after_first_comparison_prevents_remaining_jobs(self):
        self.args.follow = True
        self.args.training_pid = 42
        self.args._training_identity = "same-native-process"
        child = Mock()
        child.poll.return_value = 0
        def finish_comparison(**_kwargs):
            history.write(self.args.training / "summary.json", {"status": "failed"})
            return 0
        child.wait.side_effect = finish_comparison
        with patch.object(history, "update_sweep"), patch.object(history, "process_identity", return_value="same-native-process"), patch.object(history.subprocess, "Popen", return_value=child) as spawn:
            sweep = history.run_sweep(self.args)
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse((sweep / "finished.json").exists())
        self.assertEqual(history.read(self.args.output / "status.json")["state"], "training-failed-no-final-sweep")
        receipts = list(sweep.glob("*.receipt.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(history.read(receipts[0])["returncode"], 0)

    def test_interruption_terminates_only_the_comparison_child(self):
        child = Mock()
        child.wait.side_effect = [InterruptedError("requested stop"), 0]
        child.poll.return_value = None
        with patch.object(history.subprocess, "Popen", return_value=child):
            with self.assertRaises(InterruptedError):
                history.run_sweep(self.args)
        child.terminate.assert_called_once()
        child.kill.assert_not_called()
        self.assertTrue((self.args.training / "progress.json").exists())

    def test_follow_evaluates_exactly_one_final_snapshot_after_training_finishes(self):
        self.args.training_pid = 42
        sweeps = []
        def execute(args):
            sweep, _ = history.prepare_sweep(args)
            history.write(sweep / "finished.json", {})
            sweeps.append(sweep)
            if len(sweeps) == 1:
                history.write(args.training / "models/version-00000008.json", {"weights": [2]})
                history.write(args.training / "progress.json", {"published_version": 8})
                history.write(args.training / "summary.json", {"status": "completed-bounded-pilot"})
            return sweep
        with patch.object(history, "process_identity", return_value="same-native-process"), patch.object(history, "run_sweep", side_effect=execute) as run:
            final = history.run_follow(self.args)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(history.read(final / "manifest.json")["published_version"], 8)
        self.assertEqual(history.read(self.args.output / "status.json")["state"], "final-sweep-finished")

    def test_clean_completion_does_not_retest_same_checkpoint(self):
        self.args.training_pid = 42
        def execute(args):
            sweep, _ = history.prepare_sweep(args)
            history.write(sweep / "finished.json", {})
            history.write(args.training / "summary.json", {"status": "completed-bounded-pilot"})
            return sweep
        with patch.object(history, "process_identity", return_value="same"), patch.object(history, "run_sweep", side_effect=execute) as run:
            history.run_follow(self.args)
        self.assertEqual(run.call_count, 1)

    def test_dead_training_without_summary_does_not_start_comparisons(self):
        self.args.training_pid = 42
        with patch.object(history, "process_identity", return_value=None), patch.object(history, "run_sweep") as run:
            history.run_follow(self.args)
        run.assert_not_called()
        self.assertEqual(history.read(self.args.output / "status.json")["state"], "training-exited-without-summary")

    def test_report_revisions_preserve_old_anchor_and_are_idempotent(self):
        sweep, _ = history.prepare_sweep(self.args)
        original = {"anchor": "old", "runs": []}
        revised = {"anchor": "new", "runs": []}
        # Exercise migration from an older mutable report too.
        history.write(sweep / "report.json", original)
        old_hash = history.digest(sweep / "report.json")
        with patch.object(history, "markdown", side_effect=lambda result: result["anchor"]):
            history.publish_report(sweep, revised)
            history.publish_report(sweep, revised)
        revisions = list((sweep / "report-revisions").glob("*/report.json"))
        self.assertEqual(len(revisions), 2)
        self.assertEqual(history.read(sweep / "report-revisions" / old_hash / "report.json"), original)
        self.assertEqual(history.read(sweep / "report.json"), revised)
        pointer = history.read(sweep / "report-revision.json")
        self.assertEqual(history.digest(Path(pointer["json"])), pointer["sha256"])
        self.assertEqual(Path(pointer["markdown"]).read_text(), "new")

    def test_milestone_external_projection_joins_exact_model_and_preserves_revisions(self):
        sweep, _ = history.prepare_sweep(self.args)
        mark_paths = history.record_milestones(self.args.output, sweep, {"runs": [result(110)]})
        mark_path = Path(mark_paths[0]) / "mark.json"
        original_mark = mark_path.read_bytes()
        bridge = {"identity": {"candidate_budget": 32, "reference_budget": 512,
                              "reference_kind":"legacy-cpu-heuristic",
                              "candidate_model_sha256": "different-model"},
                  "run": "same-sweep/32-vs-512", "files_sha256": {"plan": "hash"},
                  "human_projection": {"status": "conditional-provisional-bridge", "rating": 900}}
        output = {"runs": [bridge], "human_anchor": {"sha256": "old-anchor"}}
        pending = history.milestone_projections(self.args.output, sweep, output)
        self.assertEqual(pending[0]["human_projection"]["status"], "pending-same-model-mcts512-comparison")
        bridge["identity"]["candidate_model_sha256"] = history.digest(sweep / "model.json")
        first = history.milestone_projections(self.args.output, sweep, output)
        self.assertEqual(first[0]["human_projection"]["rating"], 900)
        output["human_anchor"]["sha256"] = "new-anchor"
        bridge["human_projection"]["rating"] = 1000
        second = history.milestone_projections(self.args.output, sweep, output)
        self.assertNotEqual(first[0]["revision"], second[0]["revision"])
        self.assertEqual(history.read(Path(first[0]["revision"]))["human_projection"]["rating"], 900)
        self.assertEqual(mark_path.read_bytes(), original_mark)
        self.assertEqual(len(list((mark_path.parent / "external-projections").glob("*.json"))), 3)

    def test_learned512_cannot_become_a_historical_milestone_bridge(self):
        sweep,_=history.prepare_sweep(self.args)
        history.record_milestones(self.args.output,sweep,{"runs":[result(110)]})
        bridge={"identity":{"candidate_budget":32,"reference_budget":512,"reference_kind":"compact-model",
                            "candidate_model_sha256":history.digest(sweep/"model.json")},
                "human_projection":{"status":"conditional-provisional-bridge"}}
        output=history.milestone_projections(self.args.output,sweep,{"runs":[bridge]})
        self.assertIsNone(output[0]["bridge_run"])

    def test_corrected_rules_cannot_become_a_historical_milestone_bridge(self):
        sweep,_=history.prepare_sweep(self.args)
        history.record_milestones(self.args.output,sweep,{"runs":[result(110)]})
        bridge={"identity":{"candidate_budget":32,"reference_budget":512,"reference_kind":"legacy-cpu-heuristic",
                            "rules":history.HISTORICAL_RULES+"-v2",
                            "candidate_model_sha256":history.digest(sweep/"model.json")},
                "human_projection":{"status":"conditional-provisional-bridge"}}
        output=history.milestone_projections(self.args.output,sweep,{"runs":[bridge]})
        self.assertIsNone(output[0]["bridge_run"])

    def test_wrong_comparison_model_cannot_make_a_milestone(self):
        sweep, manifest = history.prepare_sweep(self.args)
        job = manifest["jobs"][0]
        directory = sweep / job["directory"]
        history.write(directory / "summary.json", {})
        history.write(sweep / (job["directory"] + ".receipt.json"), {"returncode": 0})
        bad = result()
        bad["identity"].update(candidate_model_sha256="wrong", workers=2, decision_limit=2048, move_ms=None)
        with patch.object(history, "report", return_value={"runs": [bad]}):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                history.update_sweep(self.args, sweep)
        self.assertFalse((self.args.output / "milestones").exists())


if __name__ == "__main__":
    unittest.main()
