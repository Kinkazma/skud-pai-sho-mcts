"""CPU-only sidecar tests: subprocesses are fakes; no Rust/Swift/GPU execution."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import paisho_teacher_bootstrap as bootstrap


def write_checkpoint(path, step=100, generation=47):
    metadata = {"formatVersion": 2, "trainingStep": step,
                "progress": {"generation": generation, "scheduler": {"completedSteps": step}}}
    body = json.dumps(metadata).encode()
    body = b"PAISHO-CKPT-V2\n" + struct.pack("<I", len(body)) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + hashlib.sha256(body).digest())


def write_snapshot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "replay.psrshard").write_bytes(b"fake shard: only subprocess protocol is under test")
    snapshot = directory / "snapshot.psrsnap"
    snapshot.write_text("PAISHO-REPLAY-SNAPSHOT\t1\nshard\t0\treplay.psrshard\tfake\t1\t2048\n")
    return snapshot


class FakeProcesses:
    def __init__(self, conclusions):
        self.conclusions = conclusions
        self.calls = []
        self.fail_teacher = False
        self.fail_train = False
        self.fail_eval = False
        self.steps_executed = 0

    def __call__(self, argv, stdout, stderr, check):
        self.calls.append(argv)
        name = Path(argv[0]).name
        options = dict(zip(argv[1::2], argv[2::2]))
        code = 0
        if "--verify" in options:
            stdout.write("verified\n")
        elif name == "teacher":
            directory = Path(options["--output"])
            if self.fail_teacher:
                self.fail_teacher = False
                directory.mkdir()
                (directory / "partial.psrshard").write_text("retain me")
                code = 9
            else:
                write_snapshot(directory)
                (directory / "provenance.json").write_text(json.dumps({"seed": int(options["--seed"]),
                    "selected_positions": int(options["--positions"])}))
        elif name == "train":
            target = int(options["--target-step"])
            selected = Path(options["--run-dir"]) / "checkpoint.ckpt"
            if not selected.exists():
                self.steps_executed += target - bootstrap.checkpoint(options["--initial-checkpoint"])["trainingStep"]
                write_checkpoint(selected, target, int(options["--generation"]))
            stdout.write(f"completed_training_step={target}\nlatest_checkpoint={selected}\n")
            if self.fail_train:
                self.fail_train = False
                code = 9
        elif name == "evaluate":
            directory = Path(options["--output-dir"])
            cycle = int(directory.parent.name.split("-")[-1])
            if self.fail_eval:
                self.fail_eval = False
                directory.mkdir()
                (directory / "partial").write_text("retain me")
                code = 9
            else:
                (directory / "result").mkdir(parents=True, exist_ok=True)
                (directory / "result/result.json").write_text(json.dumps({"analysis": {
                    "conclusion": self.conclusions[min(cycle, len(self.conclusions) - 1)]}}))
        else:
            raise AssertionError("unexpected executable: " + name)
        return subprocess.CompletedProcess(argv, code)

    def named(self, name):
        return [argv for argv in self.calls if Path(argv[0]).name == name]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.initial = self.root / "initial.ckpt"
        write_checkpoint(self.initial)
        self.source = write_snapshot(self.root / "old-replays")
        self.source2 = write_snapshot(self.root / "older-replays")
        for name in ("teacher", "train", "evaluate", "service"):
            (self.root / name).write_text("fake executable " + name)
        self.campaign = self.root / "new-campaign"
        self.args = bootstrap.parser().parse_args([
            "plan", "--campaign", str(self.campaign), "--initial-checkpoint", str(self.initial),
            "--source", str(self.source), "--source", str(self.source2.parent),
            "--service", str(self.root / "service"), "--teacher-executable", str(self.root / "teacher"),
            "--train-executable", str(self.root / "train"), "--evaluate-executable", str(self.root / "evaluate")])
        bootstrap.create_plan(self.args)

    def execute(self, fake):
        with patch.object(bootstrap.subprocess, "run", side_effect=fake):
            return bootstrap.run(self.campaign)

    def v5_plan(self):
        plan = bootstrap.read_json(self.campaign / "plan.json")
        plan.update({"champion_only": True, "minimum_teacher_cycles": 1})
        (self.campaign / "plan.json").write_text(json.dumps(plan))

    def test_v5_initial_success_still_executes_teacher_and_requires_promoted_descendant(self):
        self.v5_plan()
        fake = FakeProcesses([bootstrap.UPPER])
        with patch.object(bootstrap, "select_teacher_candidate", side_effect=lambda root, plan, incumbent, candidate, cycle: candidate):
            result = self.execute(fake)
        self.assertEqual(len(fake.named("teacher")), 1)
        self.assertEqual(result["cycles_completed"], 1)
        self.assertFalse(result["teacher_omitted"])
        self.assertTrue(result["gate_passed"])

    def test_v5_rejected_teaching_preserves_parent_and_cannot_claim_gate(self):
        self.v5_plan()
        fake = FakeProcesses([bootstrap.UPPER])
        with patch.object(bootstrap, "select_teacher_candidate", side_effect=lambda root, plan, incumbent, candidate, cycle: incumbent):
            result = self.execute(fake)
        self.assertEqual(len(fake.named("teacher")), 5)
        self.assertEqual(result["selected_checkpoint"], str(self.initial))
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["status"], "budget_exhausted")
        calls = [dict(zip(c[1::2], c[2::2])) for c in fake.named("train")]
        self.assertTrue(all(c["--initial-checkpoint"] == str(self.initial) for c in calls))

    def test_v5_selection_receipt_preserves_incumbent_on_inconclusive_and_resumes(self):
        plan = bootstrap.read_json(self.campaign / "plan.json")
        plan["binaries"]["promote"] = {"path": "promote"}
        candidate = self.root / "candidate.ckpt"
        write_checkpoint(candidate, 132, 48)
        for cycle, conclusion in enumerate(["PromoteCandidate", "RejectCandidate", "InconclusiveMaximumEligiblePairs", "InconclusiveMaximumAttemptedPairs"], 1):
            with patch.object(bootstrap, "command", return_value="conclusion=" + conclusion + "\n"):
                selected = bootstrap.select_teacher_candidate(self.campaign, plan, self.initial, candidate, cycle)
            self.assertEqual(selected, candidate if conclusion == "PromoteCandidate" else self.initial)
            with patch.object(bootstrap, "command", side_effect=AssertionError("must resume")):
                self.assertEqual(bootstrap.select_teacher_candidate(self.campaign, plan, self.initial, candidate, cycle), selected)

    def test_initial_upper_omits_teacher_and_stops_forever(self):
        fake = FakeProcesses([bootstrap.UPPER])
        result = self.execute(fake)
        self.assertTrue(result["teacher_omitted"])
        self.assertEqual(result["selected_checkpoint"], str(self.initial))
        self.assertEqual(fake.named("teacher"), [])
        before = (self.campaign / "result.json").read_bytes()
        (self.root / "service").unlink()  # Permanent exit survives later environment changes.
        with patch.object(bootstrap.subprocess, "run", side_effect=AssertionError("must stop")):
            self.assertEqual(bootstrap.run(self.campaign), result)
        self.assertEqual(before, (self.campaign / "result.json").read_bytes())

    def test_stop_at_first_upper_and_absolute_targets(self):
        fake = FakeProcesses(["supports-central-elo-window", "inconclusive-maximum-eligible-pairs", bootstrap.UPPER])
        result = self.execute(fake)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["cycles_completed"], 2)
        self.assertEqual(len(fake.named("teacher")), 2)
        training = [dict(zip(c[1::2], c[2::2])) for c in fake.named("train")]
        self.assertEqual([int(x["--target-step"]) for x in training], [132, 164])
        self.assertEqual([int(x["--generation"]) for x in training], [48, 49])
        self.assertTrue(all(x["--objective"] == "supervised" for x in training))
        self.assertEqual(training[1]["--initial-checkpoint"], str(self.campaign / "cycle-001/learner/checkpoint.ckpt"))
        count = len(fake.calls)
        self.execute(fake)
        self.assertEqual(len(fake.calls), count)

    def test_budget_exhaustion_is_not_a_win(self):
        fake = FakeProcesses(["inconclusive-maximum-attempted-pairs"])
        result = self.execute(fake)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["cycles_completed"], 5)
        self.assertEqual(fake.steps_executed, 160)
        self.assertEqual(len(fake.named("teacher")), 5)
        self.assertNotIn("terminal-ppo", [part for call in fake.calls for part in call])

    def test_partial_teacher_preserved_and_new_attempt(self):
        fake = FakeProcesses(["supports-lower-elo-hypothesis", bootstrap.UPPER])
        fake.fail_teacher = True
        with self.assertRaises(RuntimeError):
            self.execute(fake)
        partial = self.campaign / "cycle-001/teacher-attempt-000/corpus/partial.psrshard"
        self.assertEqual(partial.read_text(), "retain me")
        self.assertTrue(self.execute(fake)["gate_passed"])
        self.assertTrue(partial.exists())
        self.assertTrue((self.campaign / "cycle-001/teacher-attempt-001/corpus/snapshot.psrsnap").exists())

    def test_learner_resume_skips_committed_steps_without_relabel(self):
        fake = FakeProcesses(["supports-lower-elo-hypothesis", bootstrap.UPPER])
        fake.fail_train = True
        with self.assertRaises(RuntimeError):
            self.execute(fake)
        self.assertTrue(self.execute(fake)["gate_passed"])
        self.assertEqual(fake.steps_executed, 32)
        self.assertEqual(len(fake.named("teacher")), 1)
        self.assertEqual(fake.named("train")[0], fake.named("train")[1])

    def test_teacher_snapshot_reused_after_receipt_interruption(self):
        fake = FakeProcesses(["supports-lower-elo-hypothesis", bootstrap.UPPER])
        original = bootstrap.publish

        def interrupted(path, value):
            if Path(path).name == "teacher.json":
                raise RuntimeError("sidecar interrupted after snapshot publication")
            original(path, value)

        with patch.object(bootstrap, "publish", side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                self.execute(fake)
        self.assertTrue(self.execute(fake)["gate_passed"])
        self.assertEqual(len(fake.named("teacher")), 1)

    def test_evaluator_resumes_same_archive(self):
        fake = FakeProcesses([bootstrap.UPPER])
        fake.fail_eval = True
        with self.assertRaises(RuntimeError):
            self.execute(fake)
        self.assertTrue(self.execute(fake)["teacher_omitted"])
        attempts = [x for x in fake.named("evaluate") if "--verify" not in x]
        self.assertEqual(attempts[0], attempts[1])
        self.assertTrue((self.campaign / "evaluation-000/archive/partial").exists())

    def test_exact_source_provenance(self):
        (self.source.parent / "replay.psrshard").write_text("changed")
        with patch.object(bootstrap.subprocess, "run", side_effect=AssertionError("no child")):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                bootstrap.run(self.campaign)

    def test_new_campaign_only(self):
        with self.assertRaises(FileExistsError):
            bootstrap.create_plan(self.args)

    def test_cli_terminal_json_and_exit_zero_for_exhaustion(self):
        fake = FakeProcesses(["inconclusive-maximum-eligible-pairs"])
        out, err = io.StringIO(), io.StringIO()
        with patch.object(bootstrap.subprocess, "run", side_effect=fake), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = bootstrap.main(["run", "--campaign", str(self.campaign)])
        self.assertEqual(code, 0)
        self.assertTrue(err.getvalue())
        self.assertTrue(all(line.startswith("bootstrap_stage=") for line in err.getvalue().splitlines()))
        self.assertEqual(len(out.getvalue().splitlines()), 1)
        result = json.loads(out.getvalue())
        self.assertEqual(result, bootstrap.read_json(self.campaign / "result.json"))
        self.assertEqual(set(result), {"status", "gate_passed", "teacher_omitted", "cycles_completed",
            "selected_checkpoint", "selected_checkpoint_sha256", "evaluation", "plan_sha256",
            "teacher_stopped_permanently", "next_action"})
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["gate_passed"])

    def test_cli_failure_stderr_no_terminal_result(self):
        fake = FakeProcesses([bootstrap.UPPER])
        fake.fail_eval = True
        out, err = io.StringIO(), io.StringIO()
        with patch.object(bootstrap.subprocess, "run", side_effect=fake), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bootstrap.main(["run", "--campaign", str(self.campaign)]), 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("bootstrap:", err.getvalue())
        self.assertFalse((self.campaign / "result.json").exists())

    def test_cli_usage_exit_two(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            bootstrap.main(["run", "--campaign", str(self.campaign), "--steps", "1"])
        self.assertEqual(error.exception.code, 2)

    def test_plan_json_contract_and_worker_inheritance(self):
        args = bootstrap.parser().parse_args(["plan", "--campaign", str(self.root / "another"),
            "--source", str(self.source), "--initial-checkpoint", str(self.initial),
            "--service", str(self.root / "service"), "--eval-workers", "3"])
        args.teacher_executable = self.root / "teacher"
        args.train_executable = self.root / "train"
        args.evaluate_executable = self.root / "evaluate"
        result = bootstrap.create_plan(args)
        self.assertEqual(set(result), {"status", "campaign", "plan_sha256"})
        self.assertEqual(result["status"], "planned")
        plan = bootstrap.read_json(args.campaign / "plan.json")
        self.assertEqual(plan["evaluation"]["pairs-per-batch"], 3)

    def test_modified_plan_rejected_after_interruption(self):
        fake = FakeProcesses([bootstrap.UPPER])
        fake.fail_eval = True
        with self.assertRaises(RuntimeError):
            self.execute(fake)
        plan = bootstrap.read_json(self.campaign / "plan.json")
        plan["steps"] += 1
        (self.campaign / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "plan changed"):
            self.execute(fake)

    def test_short_checkpoint_rejected_cleanly(self):
        self.initial.write_bytes(b"PAISHO-CKPT-V2\n")
        with self.assertRaisesRegex(ValueError, "truncated"):
            bootstrap.checkpoint(self.initial)

    def test_teacher_worker_default_and_override(self):
        required = ["plan", "--campaign", "new", "--initial-checkpoint", "checkpoint",
                    "--source", "snapshot", "--service", "service"]
        for available, expected in ((10, 10), (None, 1)):
            with self.subTest(available=available), patch.object(bootstrap.os, "cpu_count", return_value=available):
                cli = bootstrap.parser()
                self.assertEqual(cli.parse_args(required).teacher_workers, expected)
                self.assertEqual(cli.parse_args(required + ["--teacher-workers", "3"]).teacher_workers, 3)

    def test_plan_defaults_and_distinct_seed_simulations(self):
        plan = bootstrap.read_json(self.campaign / "plan.json")
        self.assertEqual((plan["max_cycles"], plan["positions"], plan["steps"], plan["batch"]), (5, 2048, 32, 64))
        self.assertEqual(plan["evaluation"]["alpha"], 0.025)
        self.assertEqual(plan["evaluation"]["max-eligible-pairs"], 400)
        self.assertEqual(len(plan["sources"]), 2)
        self.assertEqual(plan["teacher_workers"], bootstrap.os.cpu_count() or 1)
        # 1000 cheap deterministic scenarios, not 1000 hardware launches.
        for master in range(1000):
            with self.subTest(seed=master):
                evaluation = {bootstrap.seed(master, d, c) for d in ("evaluation-start", "evaluation-model") for c in range(6)}
                replay = {bootstrap.seed(master, d, c) for d in ("teacher", "sampler", "model") for c in range(1, 6)}
                self.assertTrue(evaluation.isdisjoint(replay))
                self.assertEqual(len(evaluation), 12)
                self.assertEqual(bootstrap.seed(master, "teacher", 1), bootstrap.seed(master, "teacher", 1))


if __name__ == "__main__":
    unittest.main()
