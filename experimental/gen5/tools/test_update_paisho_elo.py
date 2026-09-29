#!/usr/bin/env python3
"""Refresh transaction, evidence deduplication, and version isolation regressions."""
import json
import fcntl
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import paisho_human_corpus as corpus
import update_paisho_elo as refresh

VERIFIER = corpus.ROOT / "target/release/examples/verify_record"
ARCHIVE = corpus.ROOT / "benchmarks/results/human-mcts-bridge-2026-09-08"


class BridgeIdentityTests(unittest.TestCase):
    def test_milestone_refresh_requires_legacy_kind_and_valid_projection(self):
        identity={"candidate_budget":32,"reference_budget":512,"reference_kind":"compact-model",
                  "candidate_model_sha256":"model","candidate_weights_sha256":"weights"}
        mark={"budget":32,"threshold":100,"model_sha256":"model","comparison":{"identity":identity}}
        run={"identity":identity,"complete_archive":True,"failed":False,"paired_results":{"eligible_pairs":8},
             "human_projection":{"status":"conditional-provisional-bridge"},"run":"run","files_sha256":{}}
        def bridges():
            return refresh.project_milestones([(Path("mark.json"),mark,"markhash")],
                {"runs":[run],"human_anchor":{}})["milestones"][0]["bridges"]
        self.assertEqual(bridges(),[])
        identity["reference_kind"]="legacy-cpu-heuristic"
        run["human_projection"]["status"]="withheld-incomplete-or-failed-archive"
        self.assertEqual(bridges(),[])
        run["human_projection"]["status"]="conditional-provisional-bridge"
        self.assertEqual(len(bridges()),1)


@unittest.skipUnless(VERIFIER.is_file(), "build the rules verifier first")
class RefreshTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "refresh"
        self.originals = [json.loads(path.read_text()) for path in sorted((ARCHIVE / "originals").glob("*.json"))]
        # Posterior arithmetic has independent scientific tests in test_paisho_human_corpus.
        self.patch = patch.object(corpus, "posterior", return_value={"posterior_mode": 800, "credible_95": [300, 1200]})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def put(self, name, index=0, **changes):
        data = {**self.originals[index], **changes}
        (self.source / name).write_text(json.dumps(data))

    def run_update(self, bases=()):
        return refresh.update(self.source, self.output, bases, VERIFIER, "old-test")

    def manifest(self, result):
        return json.loads((self.output / result["revision"] / "merged/manifest.json").read_text())

    def test_noop_then_new_game_keeps_immutable_previous_revision(self):
        self.put("a.json")
        first = self.run_update()
        original_summary = (self.output / first["revision"] / "summary.json").read_bytes()
        self.assertEqual(self.run_update()["status"], "unchanged")
        self.put("b.json", 1)
        second = self.run_update()
        self.assertEqual(second["calibration_eligible_games"], 2)
        self.assertNotEqual(first["revision"], second["revision"])
        self.assertEqual((self.output / first["revision"] / "summary.json").read_bytes(), original_summary)

    def test_duplicate_alias_and_base_import_count_one(self):
        self.put("a.json")
        base = self.root / "base"
        corpus.import_corpus(self.source, base, VERIFIER, "old-test")
        self.put("copy.json")
        result = self.run_update([base / "manifest.json"])
        self.assertEqual(result["source_files"], 2)
        self.assertEqual(result["calibration_eligible_games"], 1)

    def test_metadata_changes_quarantine_instead_of_overwriting_history(self):
        self.put("a.json")
        first = self.run_update()
        self.put("a.json", elo=777)
        second = self.run_update()
        self.assertEqual(second["calibration_eligible_games"], 0)
        self.assertEqual(second["quarantined_games"], 1)
        self.assertEqual(len(self.manifest(second)["sources"]), 2)
        self.assertTrue((self.output / first["revision"] / "complete.json").is_file())

    def test_removal_does_not_erase_previously_submitted_games(self):
        self.put("a.json")
        self.put("b.json", 1)
        self.run_update()
        (self.source / "a.json").unlink()
        self.assertEqual(self.run_update()["calibration_eligible_games"], 2)

    def test_malformed_file_retained_and_explicit_future_version_separated(self):
        self.put("a.json")
        self.put("b.json", 1, bot_version="learned-future-v2")
        (self.source / "broken.json").write_text("{bad json")
        result = self.run_update()
        self.assertEqual(result["rejected_source_files"], 1)
        self.assertEqual(len(result["groups"]), 2)
        old = [group for group in result["groups"] if group["bot_cohort"] == "old-test"]
        self.assertEqual(old[0]["unique_games"], 1)
        self.assertIsNone(old[0]["bot_binary_identity"])

    def test_partial_failure_does_not_publish_and_next_call_recovers(self):
        self.put("a.json")
        first = self.run_update()
        self.put("b.json", 1)
        with patch.object(refresh, "merge_manifests", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_update()
        self.assertEqual(json.loads((self.output / "latest.json").read_text())["revision"], first["revision"])
        self.assertEqual(self.run_update()["calibration_eligible_games"], 2)

    def test_modified_archive_is_rejected_even_for_unchanged_input(self):
        self.put("a.json")
        result = self.run_update()
        (self.output / result["revision"] / "calibration.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "published refresh was modified"):
            self.run_update()

    def test_external_refresh_preserves_real_internal_match_estimates(self):
        comparison = corpus.ROOT / "benchmarks/results/compact-mcts-budgets-2026-09-08/compare32-vs-old512"
        self.put("a.json")
        before = refresh.progress.read_run(comparison)
        result = refresh.update(self.source, self.output, (), VERIFIER, "old-test", [comparison])
        projected = json.loads((self.output / result["revision"] / "comparisons.json").read_text())["runs"][0]
        for key in ("paired_results", "score_equivalent_elo_delta", "regularized_pair_delta", "identity"):
            self.assertEqual(before[key], projected[key])
        self.assertEqual(projected["human_projection"]["new_human_games_for_this_model"], 0)
        self.assertEqual(projected["human_projection"]["historical_human_games"], 1)

    def test_live_history_refresh_keeps_marks_and_joins_only_exact_model(self):
        history = self.root / "history"
        sweep = history / "sweeps/sweep-000000"
        sweep.mkdir(parents=True)
        archived = corpus.ROOT / "benchmarks/results/compact-mcts-budgets-2026-09-08/compare32-vs-old512"
        comparison = sweep / "mcts32-vs-legacy512"
        shutil.copytree(archived, comparison)
        (sweep / "mcts32-vs-legacy512.receipt.json").write_text('{"returncode":0}')
        run = refresh.progress.read_run(comparison)
        original_marks = {}
        for threshold in (100, 200):
            target = history / f"milestones/mcts32/plus-{threshold:04d}"
            target.mkdir(parents=True)
            model = json.loads((comparison / "candidate.json").read_text())
            identity = dict(run["identity"])
            if threshold == 200:
                model["weights"][0] += 0.125
                model_bytes = json.dumps(model).encode()
                identity["candidate_weights_sha256"] = "different-weights"
            else:
                model_bytes = (comparison / "candidate.json").read_bytes()
            model_hash = corpus.sha256(model_bytes)
            identity["candidate_model_sha256"] = model_hash
            (target / "model.json").write_bytes(model_bytes)
            mark = {"budget": 32, "threshold": threshold, "model_sha256": model_hash,
                    "comparison": {**run, "identity": identity}}
            (target / "mark.json").write_text(json.dumps(mark))
            original_marks[target / "mark.json"] = (target / "mark.json").read_bytes()
        self.put("a.json")
        with (history / "history.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            first = refresh.update(self.source, self.output, (), VERIFIER, "old-test", histories=[history])
            self.put("b.json", 1)
            corpus.posterior.return_value = {"posterior_mode": 900, "credible_95": [400, 1300]}
            second = refresh.update(self.source, self.output, (), VERIFIER, "old-test", histories=[history])
        old = json.loads((self.output / first["milestone_projections"]).read_text())["milestones"]
        new = json.loads((self.output / second["milestone_projections"]).read_text())["milestones"]
        self.assertEqual(new[0]["status"], "conditional-provisional-bridges")
        self.assertEqual(new[1]["status"], "pending-same-model-mcts512-comparison")
        self.assertEqual(new[1]["bridges"], [])
        self.assertEqual(new[0]["internal_comparison"], old[0]["internal_comparison"])
        self.assertEqual(old[0]["bridges"][0]["human_projection"]["scenarios"][0]["anchor_mode"], 800)
        self.assertEqual(new[0]["bridges"][0]["human_projection"]["scenarios"][0]["anchor_mode"], 900)
        for path, data in original_marks.items():
            self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
