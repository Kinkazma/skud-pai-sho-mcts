import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import paisho_teacher_program as p


class TeacherProgramTests(unittest.TestCase):
    def test_new_plan_preserves_explicit_long_self_play_profile_and_disables_teacher(self):
        from types import SimpleNamespace
        binaries = self.root / "bin"
        binaries.mkdir()
        for name in ["paisho-teacher-relabel", "paisho-train", "paisho-evaluate", "paisho-promote", "paisho-live-v7", "paisho-mpsgraph-service"]:
            (binaries / name).write_text("fixture")
        options = {"external-games":"16", "actor-games":"256", "start-horizon":"128", "actor-decision-limit":"2048"}
        path = self.root / "options.json"
        path.write_text(json.dumps(options))
        args = SimpleNamespace(campaign=self.root / "new", learner=self.checkpoint,
            champion=self.checkpoint, bin_directory=binaries, live_options=path, source=[], tier="random")
        with patch.object(p.b, "checkpoint", side_effect=p.b.read_json):
            p.create(args)
        plan = p.b.read_json(args.campaign / "teacher-program-plan.json")
        for key, value in options.items():
            self.assertEqual(plan["live_options"][key], value)
        self.assertTrue(p.teacher_disabled(args.campaign))
        self.assertTrue(plan["authority"]["path"].endswith("CURRICULUM_V8.md"))

    def test_exact_requested_boundaries(self):
        for w, d, n, expected in [(60,20,100,True),(60,19,100,False),(59,30,100,False),
                                   (80,0,100,True),(79,0,100,False),(0,0,0,False),
                                   (240,80,400,True),(239,81,400,False),(320,0,400,True)]:
            self.assertEqual(p.achieved(w,d,n), expected)

    def test_unchanged_champion_reuses_verified_panel_without_new_games(self):
        evidence = self.root / "evidence.json"
        evidence.write_text("{}")
        previous = {"candidate": self.ref, "evidence": [p.b.identity(evidence)], "wins": 280}
        with patch.object(p, "measure", side_effect=AssertionError("unnecessary matches")):
            self.assertEqual(p.measure_selected(self.root, self.plan, {"last_evaluation": previous}, self.checkpoint, 1, 2), previous)
        evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            p.measure_selected(self.root, self.plan, {"last_evaluation": previous}, self.checkpoint, 1, 2)

    def test_teacher_disable_is_explicit_and_does_not_fake_goal(self):
        self.assertFalse(p.teacher_disabled(self.root))
        (self.root / "teacher-disabled.json").write_text('{"disabled": true}')
        self.assertTrue(p.teacher_disabled(self.root))
        p.write_status(self.root, self.plan, self.plan["initial_state"])
        status = json.loads((self.root / "program-status.json").read_text())
        self.assertTrue(status["teacher_disabled"])
        self.assertFalse(status["goal_passed"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.checkpoint = self.root / "parent"
        self.checkpoint.write_text(json.dumps({"trainingStep":100,"progress":{"generation":10}}))
        self.ref = p.b.identity(self.checkpoint)
        self.plan = {"initial_state":{"learner":self.ref,"champion":self.ref,"generation":10,
            "training_step":100,"teacher_steps":0,"sources":[],"tier":"random"},
            "authority":self.ref,"binaries":{},"seed":1,"chunks":2,"steps":64,"positions":4,
            "post_goal_target":0,"ppo_generations":50,"warmup_generations":10,"goal_pairs":20,
            "goal_batch_pairs":10,"goal_evaluation":{}}
        (self.root / "teacher-program-plan.json").write_text(json.dumps(self.plan))
        self.trained = []; self.live_targets = []

    def teacher(self, root, plan, chunk):
        directory = root / f"targets-{chunk}"
        directory.mkdir(exist_ok=True)
        (directory / "provenance.json").write_text(json.dumps({"teacher":{"protocol":"paisho-offline-mcts32-teacher-v1"}}))
        (directory / "snapshot").write_text("snapshot")
        return directory / "snapshot"

    def train(self, root, plan, initial, snapshot, chunk):
        path = root / f"trained-{chunk}"
        if not path.exists():
            old = p.b.read_json(initial)
            generation = max(old["progress"]["generation"],plan["generation_floor"]) + 1
            path.write_text(json.dumps({"trainingStep":old["trainingStep"]+plan["steps"],"progress":{"generation":generation}}))
            self.trained.append((str(initial),generation))
        return path

    def live(self, root, plan, state, target):
        self.live_targets.append(target)
        # Deliberately simulate rollback to an old champion; logical counter stays target.
        return {**state,"generation":target,"learner":self.ref,"training_step":100}

    def run_with(self, measure, fail_promotion=False):
        count = 0
        def promotion(root, plan, champion, candidate, chunk):
            nonlocal count
            count += 1
            if fail_promotion and count == 2: raise RuntimeError("interruption")
            return champion
        with ExitStack() as stack:
            for name, fn in [("teacher",self.teacher),("train",self.train),("checkpoint",p.b.read_json),
                             ("select_teacher_candidate",promotion)]: stack.enter_context(patch.object(p.b,name,side_effect=fn))
            stack.enter_context(patch.object(p,"measure",side_effect=measure))
            stack.enter_context(patch.object(p,"live_segment",side_effect=self.live))
            return p.execute(self.root)

    @staticmethod
    def result(passed=False, site=False):
        return {"passed":passed,"site_ready":site,"wins":80 if passed else 30,"draws":20,"games":100}

    def test_recurrence_no_occurrence_cap_and_learner_continuation(self):
        def measure(root, plan, candidate, occurrence, chunk):
            self.assertEqual(candidate, self.checkpoint)  # Never displace champion on Random gate alone.
            return self.result(occurrence==2, occurrence==2)
        result = self.run_with(measure)
        self.assertTrue(result["teacher_complete"])
        self.assertEqual(result["champion"],self.ref)
        self.assertEqual([g for _,g in self.trained],[21,22,73])
        self.assertIn("trained-1",self.trained[1][0])
        self.assertEqual(self.live_targets,[20,72])
        self.assertEqual(result["teacher_steps"],192)
        self.assertTrue((self.root / "occurrence-000001/complete.json").exists())

    def test_restart_after_partial_chunk_retains_teacher_updates(self):
        def measure(root,plan,candidate,occurrence,chunk):return self.result(occurrence==2,occurrence==2)
        with self.assertRaisesRegex(RuntimeError,"interruption"):self.run_with(measure,True)
        self.assertEqual(len(self.trained),2)
        result=self.run_with(measure)
        self.assertEqual(len(self.trained),3)
        self.assertEqual(result["teacher_steps"],192)

    def test_teacher_stops_at_60_20_but_random_ppo_waits_for_70(self):
        def measure(root,plan,candidate,occurrence,chunk):
            return self.result(occurrence>=1,occurrence>=2)
        result=self.run_with(measure)
        self.assertEqual(len(self.trained),1)
        self.assertEqual(self.live_targets,[20,71,121])
        self.assertTrue(result["site_ready"])

    def test_disabled_teacher_continues_ppo_without_supervised_updates(self):
        (self.root / "teacher-disabled.json").write_text('{"disabled": true}')
        result = self.run_with(lambda root,plan,candidate,occurrence,chunk: self.result(occurrence>=2,occurrence>=2))
        self.assertEqual(self.trained, [])
        self.assertEqual(self.live_targets, [20,70,120])
        self.assertEqual(result["teacher_steps"], 0)

    def test_already_qualified_parent_omits_teacher(self):
        result=self.run_with(lambda *args:self.result(True,True))
        self.assertEqual(self.trained,[]);self.assertEqual(self.live_targets,[])
        self.assertTrue(result["teacher_complete"])

    def test_mutated_plan_rejected_after_completion(self):
        self.run_with(lambda *args:self.result(True,True))
        self.plan["steps"] += 1
        (self.root / "teacher-program-plan.json").write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,"plan changed"):self.run_with(lambda *args:self.result(True,True))

    def test_fixed_panel_exclusions_cannot_raise_rates(self):
        plan={**self.plan,"binaries":{"evaluate":{"path":"evaluate"},"service":{"path":"service"}}}
        def command(directory,argv):
            directory.mkdir(parents=True,exist_ok=True)
            if "--verify" not in argv:
                options=dict(zip(argv[1::2],argv[2::2]));self.assertEqual(options["--opponent"],"random")
                self.assertEqual(int(options["--pairs-per-batch"]),10)
                archive=Path(options["--output-dir"]);(archive/"result").mkdir(parents=True)
                (archive/"result/result.json").write_text(json.dumps({"analysis":{
                    "attempted_pairs":10,"candidate_wins":15,"draws":0,"candidate_losses":0}}))
            return ""
        with patch.object(p.b,"command",side_effect=command): result=p.measure(self.root/"panel",plan,self.checkpoint,1,1)
        self.assertEqual(result["games"],40);self.assertEqual(result["unresolved"],10)
        self.assertEqual(result["win_rate"],.75);self.assertFalse(result["passed"])
        self.assertTrue(result["site_ready"])
        with patch.object(p.b,"command",side_effect=AssertionError("must resume")):
            self.assertEqual(p.measure(self.root/"panel",plan,self.checkpoint,1,1),result)


if __name__ == "__main__": unittest.main()
