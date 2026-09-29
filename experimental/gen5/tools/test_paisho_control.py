#!/usr/bin/env python3

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paisho_control as control


class PaishoControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="paisho-control-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(
        self,
        command: list[str] | None = None,
        target: int | None = 1024,
    ) -> control.ControlConfig:
        path = self.root / "control.json"
        payload = {
            "version": 1,
            "name": "Test Pai Sho",
            "command": command or ["/usr/bin/true"],
            "working_directory": str(self.root),
            "state_path": str(self.root / "state.json"),
            "log_path": str(self.root / "training.log"),
            "progress_directory": str(self.root / "run"),
            "target_training_step": target,
            "host": "127.0.0.1",
            "port": 18765,
            "control_token": "test-token",
            "restart_delay_seconds": 0.01,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return control.ControlConfig.load(path)

    def test_progress_counts_only_durable_committed_checkpoints(self) -> None:
        config = self.config()
        run = config.progress_directory
        assert run is not None
        run.mkdir()
        (run / "checkpoint-g0001-s00000000000000000256-a0000.psckpt").write_bytes(b"a")
        (run / "commit-s00000000000000000256.pslearn").write_bytes(b"a")
        (run / "checkpoint-g0001-s00000000000000000512-a0000.psckpt").write_bytes(b"b")
        (run / ".checkpoint-g0001-s00000000000000000768-a0000.psckpt.partial").write_bytes(b"c")

        progress = control.read_progress(config)

        self.assertEqual(progress.completed_step, 256)
        self.assertEqual(progress.checkpoint_count, 1)
        self.assertEqual(progress.percent, 25.0)

    def test_orphan_checkpoint_without_any_commit_is_not_progress(self) -> None:
        config = self.config(target=256)
        run = config.progress_directory
        assert run is not None
        run.mkdir()
        (run / "checkpoint-g1-s00000000000000000256-a0.psckpt").write_bytes(b"orphan")

        progress = control.read_progress(config)

        self.assertEqual(progress.completed_step, 0)
        self.assertEqual(progress.checkpoint_count, 0)

    def test_progress_follows_committed_checkpoints_across_generations(self) -> None:
        config = self.config(target=25_600)
        generations = config.progress_directory
        assert generations is not None
        for generation, step in ((30, 7_680), (31, 7_936)):
            learner = generations / f"generation-{generation:020}" / "learner"
            learner.mkdir(parents=True)
            checkpoint = learner / (
                f"checkpoint-g{generation:020}-s{step:020}-a{0:020}.psckpt"
            )
            checkpoint.write_bytes(b"committed")
            (learner / f"commit-s{step:020}.pslearn").write_bytes(b"commit")
        orphan = generations / f"generation-{32:020}" / "learner"
        orphan.mkdir(parents=True)
        (orphan / f"checkpoint-g{32:020}-s{8_192:020}-a{0:020}.psckpt").write_bytes(
            b"orphan"
        )

        progress = control.read_progress(config)

        self.assertEqual(progress.completed_step, 7_936)
        self.assertEqual(progress.checkpoint_count, 2)
        self.assertEqual(progress.percent, 31.0)
        self.assertIn("generation-00000000000000000031", progress.latest_checkpoint or "")

    def test_desired_state_is_atomic_and_survives_a_new_reader(self) -> None:
        config = self.config()
        control.atomic_write_json(config.state_path, control.default_state(config))

        changed = control.set_desired_without_server(config, "running")
        reloaded = control.load_state(config)

        self.assertEqual(changed["desired"], "running")
        self.assertEqual(reloaded["desired"], "running")
        self.assertGreater(reloaded["revision"], 0)
        self.assertFalse(list(self.root.glob(".*.partial-*")))

    def test_configuration_change_does_not_adopt_an_old_job_state(self) -> None:
        first = self.config(command=["/usr/bin/true"])
        control.atomic_write_json(first.state_path, control.default_state(first))
        control.set_desired_without_server(first, "running")
        second = self.config(command=["/usr/bin/false"])

        state = control.load_state(second)

        self.assertEqual(state["desired"], "paused")
        self.assertEqual(state["job_fingerprint"], second.fingerprint)

    def test_pause_resume_and_controller_restart_keep_one_process_group(self) -> None:
        config = self.config(
            command=["/usr/bin/env", "/bin/sleep", "120"],
            target=None,
        )
        control.atomic_write_json(config.state_path, control.default_state(config))
        control.set_desired_without_server(config, "running")
        first = control.Supervisor(config)
        first.reconcile()
        pid = first.child.pid if first.child else None
        self.assertIsNotNone(pid)
        assert pid is not None
        try:
            first.set_desired("paused")
            self.assertTrue(self._wait_for_process_state(pid, "T"))
            self.assertEqual(control.load_state(config)["desired"], "paused")

            restarted_controller = control.Supervisor(config)
            restarted_controller.reconcile()
            self.assertIsNone(restarted_controller.child)
            self.assertEqual(restarted_controller.status()["pid"], pid)

            restarted_controller.set_desired("running")
            self.assertTrue(self._wait_until_not_stopped(pid))
            self.assertEqual(control.load_state(config)["desired"], "running")
        finally:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if first.child:
                first.child.wait(timeout=5)

    def test_two_controllers_cannot_start_two_training_groups(self) -> None:
        config = self.config(command=["/bin/sleep", "120"], target=None)
        control.atomic_write_json(config.state_path, control.default_state(config))
        control.set_desired_without_server(config, "running")
        supervisors = [control.Supervisor(config), control.Supervisor(config)]
        barrier = threading.Barrier(2)

        def reconcile(supervisor: control.Supervisor) -> None:
            barrier.wait()
            supervisor.reconcile()

        threads = [
            threading.Thread(target=reconcile, args=(supervisor,))
            for supervisor in supervisors
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        state = control.load_state(config)
        pid = state["process_pid"]
        self.assertIsInstance(pid, int)
        self.assertEqual(sum(item.child is not None for item in supervisors), 1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for supervisor in supervisors:
            if supervisor.child:
                supervisor.child.wait(timeout=5)

    def test_child_exit_after_final_checkpoint_clears_the_pid(self) -> None:
        config = self.config(command=["/bin/sleep", "0.1"], target=256)
        control.atomic_write_json(config.state_path, control.default_state(config))
        control.set_desired_without_server(config, "running")
        supervisor = control.Supervisor(config)
        supervisor.reconcile()
        self.assertIsNotNone(supervisor.child)
        run = config.progress_directory
        assert run is not None
        run.mkdir()
        (run / "checkpoint-g1-s00000000000000000256-a0.psckpt").write_bytes(b"ok")
        (run / "commit-s00000000000000000256.pslearn").write_bytes(b"ok")
        assert supervisor.child is not None
        supervisor.child.wait(timeout=3)

        supervisor.reconcile()
        state = control.load_state(config)

        self.assertIsNone(state["process_pid"])
        self.assertEqual(state["observed"], "completed")
        self.assertTrue(state["completed"])

    def test_completed_target_prevents_a_duplicate_restart(self) -> None:
        config = self.config(command=["/usr/bin/false"], target=256)
        run = config.progress_directory
        assert run is not None
        run.mkdir()
        (run / "checkpoint-g1-s00000000000000000256-a0.psckpt").write_bytes(b"ok")
        (run / "commit-s00000000000000000256.pslearn").write_bytes(b"ok")
        state = control.default_state(config)
        state["desired"] = "running"
        control.atomic_write_json(config.state_path, state)

        supervisor = control.Supervisor(config)
        supervisor.reconcile()
        state = control.load_state(config)

        self.assertIsNone(supervisor.child)
        self.assertTrue(state["completed"])
        self.assertEqual(state["observed"], "completed")
        self.assertEqual(state["desired"], "paused")

    def test_control_page_contains_live_controls_without_exposing_a_shell(self) -> None:
        config = self.config()
        page = control.render_control_page(config).decode("utf-8")

        self.assertIn("Mettre en pause", page)
        self.assertIn("Reprendre", page)
        self.assertIn("Cette tâche précise est terminée", page)
        self.assertIn("Un diagnostic isolé", page)
        self.assertIn("if(!r.ok)throw new Error", page)
        self.assertIn("document.getElementById('dashboard').src='/dashboard?'", page)
        self.assertIn("setInterval(refresh,1000)", page)
        self.assertIn('id="generation"', page)
        self.assertIn('id="internal-elo"', page)
        self.assertNotIn("shell", page.lower())

    def test_live_metrics_read_only_small_summaries_and_reference_scale(self) -> None:
        curriculum = self.root / "curriculum"
        decisions = curriculum / "decisions"
        decisions.mkdir(parents=True)
        for sequence, generation, elo in ((0, 40, 12.5), (1, 41, 50.5)):
            payload = {
                "payload": {
                    "generation": generation,
                    "tier_before": "random",
                    "checkpoint": {"sha256": f"checkpoint-{generation}"},
                    "evidence": {
                        "kind": "fixed-opponent",
                        "contextual_elo_point": {"kind": "finite", "elo": elo},
                        "davidson_candidate_minus_opponent": {
                            "estimate": elo + 10,
                            "paired_cluster": {
                                "interval_95_lower": elo - 20,
                                "interval_95_upper": elo + 20,
                            },
                        },
                        "eligible_pairs": 64,
                        "conclusion": "supports-central-elo-window",
                    },
                }
            }
            (decisions / f"decision-{sequence:020}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        ratings = self.root / "ratings.tsv"
        ratings.write_text(
            "PAISHO-INTERNAL-RATINGS\t1\n"
            "alias\tagent_id\telo\tmodel_se\tmodel_ci95_low\tmodel_ci95_high\t"
            "paired_cluster_se\tpaired_cluster_ci95_low\tpaired_cluster_ci95_high\t"
            "rated_games\trating_status\n"
            "site-bot-v1\tsha256:abc\t680.5\t1\t670\t690\t2\t660\t700\t72\tprovisional\n",
            encoding="utf-8",
        )
        dashboard = control.DashboardConfig(
            html_path=self.root / "dashboard.html",
            refresh_command=None,
            refresh_working_directory=None,
            minimum_refresh_seconds=60,
            curriculum_directory=curriculum,
            internal_ratings_path=ratings,
        )
        reader = control.LiveCampaignMetrics(dashboard)
        progress = control.Progress(
            completed_step=10_496,
            target_step=25_600,
            checkpoint_count=41,
            latest_checkpoint=str(
                self.root
                / "checkpoint-g00000000000000000041-s00000000000000010496-a00000000000000000000.psckpt"
            ),
        )

        metrics = reader.read(progress)

        self.assertEqual(metrics["active_generation"], 41)
        self.assertEqual(metrics["latest_evaluation"]["elo_gap"], 50.5)
        self.assertEqual(metrics["best_comparable_evaluation"]["generation"], 41)
        self.assertEqual(metrics["rating_references"][0]["elo"], 680.5)
        self.assertIsNone(metrics["internal_rating"])
        self.assertEqual(metrics["internal_rating_status"], "bridge-pending")

    @staticmethod
    def _process_state(pid: int) -> str:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "state="],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _wait_for_process_state(self, pid: int, prefix: str) -> bool:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self._process_state(pid).startswith(prefix):
                return True
            time.sleep(0.02)
        return False

    def _wait_until_not_stopped(self, pid: int) -> bool:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self._process_state(pid)
            if state and not state.startswith("T"):
                return True
            time.sleep(0.02)
        return False


if __name__ == "__main__":
    unittest.main()
