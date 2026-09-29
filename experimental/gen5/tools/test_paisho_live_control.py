"""CPU-only live-layout adapter tests; no service or controller is started."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import paisho_control as control
from tools.paisho_live_control import LiveLayoutReader


class LiveControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="paisho-live-control-test-")
        self.root = Path(self.temporary.name)
        self.reader = LiveLayoutReader(self.root)
        self.write("live-plan.json", {
            "format": "paisho-live-campaign-v1", "durable_every": 5,
            "options": {"promotion_every": 10, "evaluation_every": 10},
        })

    def tearDown(self):
        self.temporary.cleanup()

    def test_collection_results_survive_next_collection_without_replay_reads(self):
        results = {"generation": 115, "opponent": "random", "wins": 100,
                   "losses": 300, "draws": 112, "games": 512, "attempts": 576}
        self.write("live-status.json", {"generation": 115, "collection_results": results})
        self.assertEqual(self.reader.read()["collection_results"], results)
        self.write("live-status.json", {"generation": 116, "phase": "actors"})
        self.assertEqual(self.reader.read()["collection_results"], results)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(path)

    def block(self, generation=10, step=1000):
        block = {
            "generation": generation, "checkpoint": f"/not-opened/g{generation}.psckpt",
            "checkpoint_sha256": f"sha-{generation}", "training_step": step,
            "champion": "/not-opened/champion.psckpt", "tier": "random",
            "games": generation * 512, "examples": generation * 6400,
            "attempt_directory": "/not-traversed/attempt",
        }
        self.write(f"blocks/block-{generation:020}.json", block)
        return block

    def assessment(self, generation=10, point=None):
        self.write(f"assessments/assessment-{generation:020}.json", {
            "generation": generation, "candidate_sha256": f"sha-{generation}",
            "selected": f"/not-opened/g{generation}.psckpt", "champion": "/not-opened/champion.psckpt",
            "tier": "site-bot-v1", "promotion": "PromoteCandidate",
            "evaluation": {
                "contextual_elo_point": point or {"kind": "finite", "elo": 112.5},
                "mle": {"candidate_minus_opponent": {"estimate": 120.0, "paired_cluster": {
                    "interval_95_lower": 80.0, "interval_95_upper": 160.0,
                }}},
                "eligible_pairs": 64, "conclusion": "supports-higher-elo",
            },
        })

    def test_ram_progress_does_not_become_durable_and_only_json_is_read(self):
        self.block()
        self.assessment()
        self.write("live-status.json", {
            "format": "paisho-live-status-v1", "phase": "actors", "generation": 13,
            "completed_generation": 12, "durable_generation": 10,
            "games": 6144, "examples": 76800,
        })
        reads = []
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            self.assertEqual(path.suffix, ".json")
            self.assertTrue(path.is_relative_to(self.root))
            reads.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text), patch("subprocess.run", side_effect=AssertionError("no processes")):
            summary = self.reader.read()
            count = len(reads)
            self.assertEqual(self.reader.read(), summary)
            self.assertEqual(len(reads), count)  # Unchanged JSON is cached.
        self.assertEqual((summary["current_generation"], summary["completed_generation"], summary["durable_generation"]), (13, 12, 10))
        self.assertEqual(summary["completed_step"], 1000)
        self.assertEqual((summary["games"], summary["examples"]), (6144, 76800))
        latest = summary["latest_evaluation"]
        self.assertEqual(latest["opponent"], "random")  # Not the post-evaluation tier.
        self.assertEqual(latest["elo_gap"], 112.5)
        self.assertEqual(latest["ci95_low"], 80.0)

    def test_live_has_no_adam_target_even_after_assessment(self):
        self.block()
        config = SimpleNamespace(progress_directory=self.root, target_training_step=1000)
        progress = control.read_progress(config)
        self.assertEqual(progress.completed_step, 1000)
        self.assertIsNone(progress.target_step)
        self.assertIsNone(progress.percent)
        self.assertFalse(progress.target_reached)
        self.assertTrue(progress.live["assessment_pending"])
        self.assessment()
        progress = control.read_progress(config)
        self.assertFalse(progress.target_reached)
        self.assertEqual(control.LiveCampaignMetrics(None).read(progress)["durable_generation"], 10)

    def test_console_has_separate_live_generation_and_total_counters(self):
        page = control.render_control_page(SimpleNamespace(name="Live test", control_token="test", dashboard=None)).decode()
        for identifier in ("live-current", "live-completed", "live-durable", "live-games", "live-examples"):
            self.assertIn(f'id="{identifier}"', page)
        self.assertIn('id="live-metrics" class="panel" hidden', page)
        self.assertIn("document.getElementById('live-metrics').hidden=m.layout!=='paisho-live'", page)

    def test_status_missing_or_invalid_falls_back_to_latest_committed_block(self):
        self.block(5, 500)
        self.write("blocks/block-00000000000000000009.pending-123.json", {"generation": 9, "training_step": 900})
        for content in (None, "{unfinished"):
            if content is not None:
                (self.root / "live-status.json").write_text(content, encoding="utf-8")
            summary = self.reader.read()
            self.assertEqual(summary["completed_step"], 500)
            self.assertEqual(summary["durable_generation"], 5)
            self.assertEqual(summary["completed_generation"], 5)
            self.assertEqual(summary["checkpoint_count"], 1)
            self.assertEqual(summary["games"], 2560)
        self.write("live-status.json", {"generation": 4, "completed_generation": 3, "games": 12})
        self.assertEqual(self.reader.read()["games"], 2560)

    def test_latest_evaluation_survives_unmeasured_blocks_and_preserves_infinity(self):
        self.block()
        self.assessment(point={"kind": "positive-infinity"})
        self.block(15, 1500)
        self.write("assessments/assessment-00000000000000000015.json", {
            "generation": 15, "candidate_sha256": "sha-15", "evaluation": None,
        })
        latest = self.reader.read()["latest_evaluation"]
        self.assertEqual(latest["generation"], 10)
        self.assertIsNone(latest["elo_gap"])
        self.assertEqual(latest["elo_kind"], "positive-infinity")
        self.assertFalse(self.reader.read()["assessment_pending"])

    def test_status_without_block_is_not_a_checkpoint_commit(self):
        self.write("live-status.json", {"generation": 34, "completed_generation": 33, "durable_generation": 30, "games": 1536, "examples": 19200})
        summary = self.reader.read()
        self.assertEqual(summary["checkpoint_count"], 0)
        self.assertEqual(summary["completed_step"], 0)
        self.assertIsNone(summary["latest_checkpoint"])

    def test_best_uses_same_opponent_and_protocol_excluding_seeds(self):
        self.block(10, 1000)
        self.assessment(10, {"kind": "finite", "elo": 90.0})
        other = self.block(20, 2000)
        other["tier"] = "site-bot-v1"
        self.write("blocks/block-00000000000000000020.json", other)
        self.assessment(20, {"kind": "finite", "elo": 500.0})
        self.block(30, 3000)
        self.assessment(30, {"kind": "finite", "elo": -20.0})
        metrics = self.reader.read()
        self.assertEqual(metrics["latest_evaluation"]["generation"], 30)
        self.assertEqual(metrics["best_comparable_evaluation"]["generation"], 10)
        first = {"curriculum_sampling_temperature": 1.0, "curriculum_start_seed": 1}
        second = {**first, "curriculum_start_seed": 2}
        self.assertEqual(self.reader._protocol(first), self.reader._protocol(second))
        second["curriculum_sampling_temperature"] = None
        self.assertNotEqual(self.reader._protocol(first), self.reader._protocol(second))

    def test_g50_numeric_schema_with_absent_cluster_interval(self):
        self.block(50, 12035)
        self.assessment(50, {"kind": "finite", "elo": -136.9690723288825})
        path = self.root / "assessments/assessment-00000000000000000050.json"
        assessment = json.loads(path.read_text())
        assessment["evaluation"]["eligible_pairs"] = 4
        assessment["evaluation"]["mle"]["candidate_minus_opponent"] = {
            "estimate": -249.82726017870033,
            "model": {"interval_95_lower": -641.8210273600462, "interval_95_upper": 142.16650700264546},
            "paired_cluster": None,
        }
        self.write(str(path.relative_to(self.root)), assessment)
        latest = self.reader.read()["latest_evaluation"]
        self.assertEqual(latest["opponent"], "random")
        self.assertAlmostEqual(latest["elo_gap"], -136.9690723288825)
        self.assertAlmostEqual(latest["davidson_gap"], -249.82726017870033)
        self.assertEqual(latest["eligible_pairs"], 4)
        self.assertIsNone(latest["ci95_low"])
        self.assertIsNone(latest["ci95_high"])

    def test_legacy_dispatch_is_unchanged_without_live_plan(self):
        (self.root / "live-plan.json").unlink()
        (self.root / "checkpoint-g1-s00000000000000000256-a0.psckpt").write_bytes(b"not-read")
        (self.root / "commit-s00000000000000000256.pslearn").write_bytes(b"not-read")
        progress = control.read_progress(SimpleNamespace(progress_directory=self.root, target_training_step=256))
        self.assertEqual(progress.completed_step, 256)
        self.assertIsNone(progress.live)
        self.assertTrue(progress.target_reached)
        self.assertIsNone(self.reader.read())

    def test_teacher_program_reads_summary_without_weights_or_replay(self):
        self.write("teacher-program-plan.json", {"format": "teacher-program-v1"})
        self.write("program-status.json", {
            "phase": "teacher-learning", "occurrence": 2, "chunk": 3,
            "chunks_per_occurrence": 10, "steps_per_chunk": 6400,
            "teacher_steps_completed": 76800, "generation": 193,
            "durable_generation": 193, "completed_step": 90000,
            "checkpoint": "/not-opened/teacher.psckpt",
            "champion_checkpoint": "/not-opened/champion.psckpt",
            "learner_run_directory": "/not-traversed/learner",
            "last_evaluation": {"wins": 55, "draws": 25, "losses": 20,
                                "games": 100, "passed": False},
            "goal_passed": False,
        })
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            self.assertIn(path.name, ("teacher-program-plan.json", "program-status.json"))
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            progress = control.read_progress(SimpleNamespace(
                progress_directory=self.root, target_training_step=6400))
        self.assertEqual(progress.completed_step, 90000)
        self.assertIsNone(progress.target_step)
        self.assertFalse(progress.target_reached)
        self.assertEqual(progress.live["phase"], "teacher-learning")
        self.assertEqual(progress.live["teacher_steps_completed"], 76800)
        self.assertFalse(progress.live["teacher_goal_passed"])

    def test_teacher_ppo_forwards_current_live_status_only_during_ppo(self):
        self.write("teacher-program-plan.json", {})
        self.write("active/live-plan.json", {"options": {}})
        self.write("active/live-status.json", {
            "generation": 210, "completed_generation": 209, "durable_generation": 205,
            "phase": "actors", "games": 8192,
        })
        status = {"phase": "ppo", "generation": 200, "durable_generation": 200,
                  "active_live_directory": "active", "occurrence": 1}
        self.write("program-status.json", status)
        summary = self.reader.read()
        self.assertEqual(summary["current_generation"], 210)
        self.assertEqual(summary["phase"], "ppo")
        self.assertEqual(summary["games"], 8192)
        self.assertTrue(summary["teacher_program"])
        self.write("program-status.json", {**status, "phase": "teacher-targets"})
        summary = self.reader.read()
        self.assertEqual(summary["current_generation"], 200)
        self.assertIsNone(summary["games"])

    def test_new_ppo_segment_retains_imported_durable_checkpoint_and_elo(self):
        self.write("teacher-program-plan.json", {})
        checkpoint = str(self.root / "previous/attempts/a/generation-00000000000000000235/checkpoint.psckpt")
        self.write("program-status.json", {"phase": "ppo", "generation": 235,
            "durable_generation": 235, "completed_step": 121600,
            "checkpoint": checkpoint, "active_live_directory": "active"})
        self.write("active/live-plan.json", {"options": {}})
        self.write("active/live-status.json", {"generation": 239,
            "completed_generation": 238, "durable_generation": 235})
        self.write("previous/live-plan.json", {"options": {}})
        self.write("previous/blocks/block-00000000000000000230.json", {
            "generation": 230, "checkpoint_sha256": "abc", "tier": "random"})
        self.write("previous/assessments/assessment-00000000000000000230.json", {
            "generation": 230, "candidate_sha256": "abc",
            "evaluation": {"contextual_elo_point": {"kind": "finite", "elo": -140.35}}})
        result = self.reader.read()
        self.assertEqual(result["completed_step"], 121600)
        self.assertEqual(result["checkpoint_count"], 1)
        self.assertEqual(result["latest_checkpoint"], checkpoint)
        self.assertEqual(result["completed_generation"], 238)
        self.assertEqual(result["latest_evaluation"]["elo_gap"], -140.35)
        self.assertTrue(result["latest_evaluation"]["inherited"])
        self.write("active/blocks/block-00000000000000000240.json", {
            "generation": 240, "training_step": 122880, "checkpoint": "new.psckpt"})
        result = self.reader.read()
        self.assertEqual(result["completed_step"], 122880)
        self.assertEqual(result["latest_checkpoint"], "new.psckpt")
        self.assertEqual(result["checkpoint_count"], 2)

    def test_teacher_program_before_first_status_has_no_false_completion(self):
        self.write("teacher-program-plan.json", {})
        summary = self.reader.read()
        self.assertEqual(summary["phase"], "starting")
        self.assertFalse(summary["teacher_goal_passed"])
        self.assertEqual(summary["checkpoint_count"], 0)
        self.assertIsNone(summary["active_generation"])

    def test_teacher_console_names_stages_and_correct_random_thresholds(self):
        page = control.render_control_page(SimpleNamespace(
            name="Teacher test", control_token="test", dashboard=None)).decode()
        self.assertIn('id="teacher-panel"', page)
        self.assertIn("Apprentissage avec le professeur", page)
        self.assertIn("Préparation des cibles du professeur", page)
        self.assertIn("60 % de victoires et 20 % de nulles", page)
        self.assertIn("80 % de victoires", page)
        self.assertIn("Objectif contre l’aléatoire", page)


if __name__ == "__main__":
    unittest.main()
