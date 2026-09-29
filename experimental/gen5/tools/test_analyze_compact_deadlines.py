import json
from pathlib import Path
import tempfile
import unittest

import analyze_compact_deadlines as analysis


def row(identity, seconds, disposition="terminal", decisions=80, roots=40, samples=32,
        cap=512, archive_cap=32):
    return analysis.Observation(str(identity),seconds,disposition,decisions,roots,samples,
                                cap,disposition,"guest" if disposition=="terminal" else None,archive_cap)


class ProjectionTests(unittest.TestCase):
    def test_known_trace_cutoff_charges_rejections_and_preserves_uncapped_volume(self):
        result=analysis.estimate([row(1,2,decisions=40,roots=20,samples=20),
                                  row(2,4,decisions=160,roots=80),row(3,6,"time-censored")],4)
        self.assertTrue(result["point_estimate_identified"])
        self.assertEqual(result["occupied_worker_seconds"],10)
        self.assertEqual(result["known_completed_games"],2)
        self.assertAlmostEqual(result["rates"]["terminal_games"]["per_occupied_worker_second"],0.2)
        self.assertAlmostEqual(result["rates"]["potential_terminal_examples_capped"]["per_occupied_worker_second"],5.2)
        self.assertEqual(result["rates"]["retained_learner_roots"]["per_occupied_worker_second"],10)
        self.assertEqual(result["rates"]["retained_psr_decisions"]["per_occupied_worker_second"],20)
        self.assertEqual(result["rejected_compute_fraction"],0.4)
        self.assertEqual(result["retained_psr_length"]["known_retained_median"],100)

    def test_global_censor_before_proposed_cutoff_produces_bounds_not_point_rate(self):
        result=analysis.estimate([row(1,2),row(2,3,"time-censored")],6)
        self.assertFalse(result["point_estimate_identified"])
        self.assertIsNone(result["occupied_worker_seconds"])
        self.assertEqual(result["occupied_worker_seconds_bounds"],[5,8])
        self.assertIsNone(result["acceptance"])
        self.assertEqual(result["acceptance_bounds"],[0.5,1])
        self.assertEqual(result["rates"]["terminal_games"]["rate_bounds"],[1/8,2/5])
        self.assertIsNone(result["rates"]["terminal_games"]["per_occupied_worker_second"])
        self.assertEqual(result["rejected_compute_fraction_bounds"],[0,0.75])
        self.assertEqual(analysis.pareto_front([result]),[])

    def test_decision_limit_is_final_rejection_under_same_decision_protocol(self):
        result=analysis.estimate([row(1,2),row(2,3,"decision-limit")],6)
        self.assertTrue(result["point_estimate_identified"])
        self.assertEqual(result["known_completed_games"],1)
        self.assertEqual(result["unknown_beyond_censoring"],0)
        self.assertEqual(result["occupied_worker_seconds"],5)
        self.assertEqual(result["rates"]["terminal_games"]["per_occupied_worker_second"],0.2)
        self.assertEqual(result["rejected_compute_fraction"],0.6)

    def test_blocked_position_never_gains_a_terminal_outcome_with_more_time(self):
        result=analysis.estimate([row(1,2),row(2,3,"no-legal-actions")],60)
        self.assertEqual(result["known_completed_games"],1)
        self.assertEqual(result["unknown_beyond_censoring"],0)
        self.assertEqual(result["occupied_worker_seconds"],5)

    def test_truncation_can_never_become_a_draw_or_generate_known_examples(self):
        for disposition in ("time-censored","decision-limit","no-legal-actions","error"):
            result=analysis.estimate([row(1,8,disposition)],4)
            self.assertEqual(result["known_completed_games"],0)
            self.assertEqual(result["rates"]["retained_learner_roots"]["known_retained"],0)
            self.assertEqual(result["rates"]["potential_terminal_examples_capped"]["known_retained"],0)
            self.assertEqual(result["rejected_compute_fraction"],1)

    def test_same_capped_examples_can_hide_real_longer_game_volume(self):
        short=analysis.estimate([row(1,4,decisions=80,roots=40)],10)
        long=analysis.estimate([row(2,4,decisions=200,roots=100)],10)
        self.assertEqual(short["rates"]["potential_terminal_examples_capped"],long["rates"]["potential_terminal_examples_capped"])
        self.assertGreater(long["rates"]["retained_learner_roots"]["per_occupied_worker_second"],
                           short["rates"]["retained_learner_roots"]["per_occupied_worker_second"])

    def test_archive_bounds_use_producer_capacity_not_analysis_cap(self):
        result=analysis.estimate([row(1,3,"time-censored",decisions=160,roots=80,samples=64,archive_cap=64)],6,sample_cap=32)
        self.assertEqual(result["rates"]["archived_root_samples"]["possible_retained_upper"],64)
        self.assertEqual(result["rates"]["potential_terminal_examples_capped"]["possible_retained_upper"],32)
        compare=analysis.estimate([row(1,3,"time-censored",samples=0,archive_cap=0)],6)
        self.assertEqual(compare["rates"]["archived_root_samples"]["possible_retained_upper"],0)

    def test_pareto_does_not_choose_only_the_fastest_game_count(self):
        traces=[row(1,1,decisions=80,roots=40),row(2,10,decisions=512,roots=512)]
        results=[analysis.estimate(traces,cutoff) for cutoff in (0.5,1,10,20)]
        # 1s gives more games/s; 10s preserves more roots/s and wastes no work.
        self.assertEqual(analysis.pareto_front(results),[1,10,20])
        errors=analysis.estimate([row(1,1),row(2,2,"error")],3)
        self.assertEqual(analysis.pareto_front([errors]),[])

    def test_grid_is_budget_specific_and_does_not_invent_heavier_data(self):
        traces=[row(1,2),row(2,4)]
        self.assertIn(8,analysis.propose_grid(traces,32))
        self.assertNotIn(8,analysis.propose_grid(traces,8))
        self.assertEqual(analysis.propose_grid([],128),[])

    def test_reject_invalid_deadlines(self):
        for deadline in (0,-1,float("inf"),float("nan")):
            with self.assertRaises(ValueError):analysis.estimate([row(1,2)],deadline)


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.psr=(Path(__file__).resolve().parents[1]/"benchmarks/results/human-mcts-bridge-2026-09-07/human.psr").read_text()

    def write_json(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def selfplay(self,name="run", weights=None):
        root=self.root/name
        model={"feature_schema":"paisho-compact-value-features-v1","weights":weights or [0.0]*64}
        self.write_json(root/"models/version-00000000.json",model)
        plan={"schema":"paisho-compact-selfplay-plan-v1","build_source_sha256":"source-a","workers":2,
              "options":{"decision_limit":512,"simulations":8,"samples":32,"learn":False},
              "search":{"simulations":8}}
        self.write_json(root/"plan.json",plan)
        game={"schema":"paisho-compact-selfplay-game-v1","termination":"terminal","outcome":"guest",
              "error":None,"elapsed_seconds":2,"record_path":"game-00000000.psr","eligible_learner_roots":28,
              "sampled_roots":[{}]*28,"simulations_requested_per_decision":8,"snapshot_version":0,
              "weights_sha256":analysis.model_hash(root/"models/version-00000000.json"),
              "build_source_sha256":"source-a","self_play":True,"learner_seat":None,
              "eligible_for_terminal_training":True,"decisions":28,"record_sha256":analysis.digest(self.psr.encode())}
        self.write_json(root/"games/game-00000000.json",game)
        (root/"games/game-00000000.psr").write_text(self.psr)
        return root,game

    def test_real_saved_psr_count_and_model_hash_are_verified(self):
        root,game=self.selfplay()
        result=analysis.analyze([root],deadlines=[1,2,8])
        self.assertEqual(len(result["groups"]),1)
        self.assertEqual(result["groups"][0]["observed"]["terminal_psr_decisions"],28)
        self.assertIn("aucun seuil",analysis.report_markdown(result))
        game["record_sha256"]="bad"
        self.write_json(root/"games/game-00000000.json",game)
        with self.assertRaisesRegex(ValueError,"PSR hash mismatch"):analysis.load_run(root)

    def test_different_model_snapshots_and_modes_never_pool(self):
        root,game=self.selfplay()
        self.write_json(root/"models/version-00000001.json",{
            "feature_schema":"paisho-compact-value-features-v1","weights":[1.0]+[0.0]*63})
        for index,(version,self_play,seat) in enumerate(((1,True,None),(0,False,"host"),(0,False,"guest")),1):
            changed={**game,"snapshot_version":version,"weights_sha256":analysis.model_hash(root/f"models/version-{version:08}.json"),
                     "self_play":self_play,"learner_seat":seat,"record_path":f"game-{index:08}.psr"}
            self.write_json(root/f"games/game-{index:08}.json",changed)
            (root/f"games/game-{index:08}.psr").write_text(self.psr)
        result=analysis.analyze([root,root],deadlines=[3])
        self.assertEqual(len(result["groups"]),4)
        self.assertEqual(sum(g["observed"]["attempted_games"] for g in result["groups"]),4)
        self.assertEqual(len(result["run_aggregates"]),2)  # Distinct weights never pool.
        self.assertEqual(sorted(g["observed"]["attempted_games"] for g in result["run_aggregates"]),[1,3])
        self.assertEqual(sorted(len(g["mode_breakdown"]) for g in result["run_aggregates"]),[1,3])

    def test_unknown_schema_and_model_mismatch_fail(self):
        root,game=self.selfplay()
        game["weights_sha256"]="different"
        self.write_json(root/"games/game-00000000.json",game)
        with self.assertRaisesRegex(ValueError,"frozen model version"):analysis.load_run(root)
        self.write_json(root/"plan.json",{"schema":"another-protocol"})
        with self.assertRaisesRegex(ValueError,"unsupported compact plan"):analysis.load_run(root)

    def test_mixed_budget_repetition_loss_stays_a_nonterminal_fixed_protocol_stop(self):
        root,game=self.selfplay()
        plan=json.loads((root/"plan.json").read_text())
        plan["options"].update(budgets=[8,32],repetition_cycles=4)
        plan["worker_assignments"]=[{"worker":0,"simulations":8},{"worker":1,"simulations":32}]
        self.write_json(root/"plan.json",plan)
        game.update(worker=1,simulations_requested_per_decision=32,termination="repetition-training-loss",
                    outcome="ongoing",eligible_for_terminal_training=False,
                    eligible_for_repetition_training=True,training_adjudication={"loser":"guest"})
        self.write_json(root/"games/game-00000000.json",game)
        result=analysis.analyze([root],deadlines=[8])
        group=result["groups"][0]
        self.assertEqual(group["identity"]["budget"],32)
        self.assertEqual(group["identity"]["repetition_cycles"],4)
        self.assertEqual(group["observed"]["dispositions"],{"repetition-adjudicated":1})
        self.assertEqual(group["estimates"][0]["known_completed_games"],0)
        self.assertTrue(group["estimates"][0]["point_estimate_identified"])

    def test_comparison_ignores_unplayed_and_never_claims_archived_q_samples(self):
        root=self.root/"comparison"
        self.write_json(root/"candidate.json",{"feature_schema":"paisho-compact-value-features-v1","weights":[0.0]*64})
        self.write_json(root/"plan.json",{"schema":"paisho-compact-comparison-plan-v1",
            "reference":analysis.LEGACY_COMPARISON_REFERENCE,
            "candidate":{"snapshot":"candidate.json","sha256":analysis.digest((root/"candidate.json").read_bytes())},
            "build":{"source_sha256":"source-a"},"decision_limit":512,"mcts":{"simulations":32},"workers":2})
        game={"schema":"paisho-compact-comparison-game-v1","termination":{"kind":"rules","outcome":"guest_win"},
              "wall_seconds":2,"candidate_host":True,"record":"records/game-00000000.psr","decisions":28,
              "candidate":{"completed_decisions":14},"reference":{"completed_decisions":14},
              "record_sha256":analysis.digest(self.psr.encode())}
        self.write_json(root/"games/game-00000000.json",game)
        (root/"records").mkdir();(root/"records/game-00000000.psr").write_text(self.psr)
        self.write_json(root/"games/game-00000001.json",{**game,"termination":{"kind":"not_played"},
            "wall_seconds":0,"decisions":0,"record":None})
        result=analysis.analyze([root],deadlines=[3])
        self.assertEqual(len(result["inputs"][0]["not_played"]),1)
        estimate=result["groups"][0]["estimates"][0]
        self.assertEqual(estimate["rates"]["archived_root_samples"]["known_retained"],0)
        self.assertEqual(estimate["rates"]["potential_terminal_examples_capped"]["known_retained"],14)
        self.assertEqual(result["groups"][0]["identity"]["reference_budget"],32)
        plan=json.loads((root/"plan.json").read_text())
        plan["reference_simulations"]=512
        self.write_json(root/"plan.json",plan)
        changed=analysis.analyze([root],deadlines=[3])
        self.assertEqual(changed["groups"][0]["identity"]["reference_budget"],512)
        self.assertEqual(changed["groups"][0]["identity"]["mode"],"vs-legacy")
        self.write_json(root/"reference.json",{"feature_schema":"paisho-compact-value-features-v1","weights":[0.5]*64})
        plan.update(reference_kind="compact-model",reference="frozen learned parent",
                    reference_model={"snapshot":"reference.json","sha256":analysis.digest((root/"reference.json").read_bytes())})
        self.write_json(root/"plan.json",plan)
        learned=analysis.analyze([root],deadlines=[3])
        self.assertEqual(learned["groups"][0]["identity"]["mode"],"vs-compact-reference")
        self.assertEqual(learned["groups"][0]["identity"]["reference_weights_sha256"],analysis.model_hash(root/"reference.json"))
        (root/"reference.json").write_text("changed")
        with self.assertRaisesRegex(ValueError,"reference snapshot hash mismatch"):analysis.load_run(root)


if __name__=="__main__":
    unittest.main()
