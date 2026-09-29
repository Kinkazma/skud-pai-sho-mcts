import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_compact_progress as progress
from analyze_compact_deadlines import LEGACY_COMPARISON_REFERENCE


def game(pair,leg,outcome="draw",kind="rules",decisions=28,turns=21):
    return {"schema":"paisho-compact-comparison-game-v1","pair_id":pair,"leg":leg,
            "game_index":2*(pair-100)+leg,"candidate_host":leg==0,
            "starting_flower":"R3","host_seed":"host","guest_seed":"guest",
            "termination":{"kind":kind,**({"outcome":outcome} if kind=="rules" else {})},
            "decisions":decisions,"completed_turns":turns,"record":"game.psr"}


class PairTests(unittest.TestCase):
    def test_score_is_not_win_rate(self):
        result=progress.wdl([1,0.5,0.5,0])
        self.assertEqual(result["win_rate"],0.25)
        self.assertEqual(result["draw_rate"],0.5)
        self.assertEqual(result["loss_rate"],0.25)
        self.assertEqual(result["score"],0.5)
        self.assertEqual(progress.expected_score_elo(result["score"])["value"],0)

    def test_expected_score_delta_and_boundaries(self):
        self.assertAlmostEqual(progress.expected_score_elo(0.75)["value"],400*math.log10(3))
        self.assertEqual(progress.expected_score_elo(1),{"value":None,"boundary":"positive-infinity"})
        self.assertEqual(progress.expected_score_elo(0),{"value":None,"boundary":"negative-infinity"})
        self.assertEqual(progress.expected_score_elo(None)["boundary"],"no-eligible-pairs")

    def test_sign_test_matches_project_reference(self):
        self.assertAlmostEqual(progress.sign_test(10,0),0.001953125)
        self.assertAlmostEqual(progress.sign_test(8,2),0.109375)
        self.assertEqual(progress.sign_test(0,0),1)
        self.assertEqual(progress.sign_test(5,5),1)

    def test_incomplete_pair_excludes_its_known_terminal_from_main_score(self):
        games=[game(100,0,"host_win"),game(100,1,"draw"),
               game(101,0,"host_win"),game(101,1,kind="wall_limit")]
        result=progress.pair_results(games,100,2)
        self.assertEqual(result["eligible_pairs"],1)
        self.assertEqual(result["excluded_pairs"],1)
        self.assertEqual(result["paired_wdl"]["score"],0.75)
        self.assertEqual(result["paired_wdl"]["wins"],1)
        self.assertEqual(result["all_terminal_wdl"]["wins"],2)
        self.assertEqual(result["scheduled_score_missing_outcome_bounds"],[0.625,0.875])
        self.assertEqual(result["eligible_lengths_by_result"]["win"]["decisions"]["count"],1)
        self.assertEqual(result["eligible_lengths_by_result"]["loss"]["decisions"]["count"],0)

    def test_no_legal_actions_never_supply_a_score(self):
        games=[game(100,0,kind="no_legal_actions"),game(100,1,"host_win")]
        result=progress.pair_results(games,100,1)
        self.assertEqual(result["eligible_pairs"],0)
        self.assertEqual(result["all_terminal_wdl"]["games"],1)
        self.assertEqual(result["all_terminal_wdl"]["draws"],0)
        self.assertIn("no_legal_actions",result["excluded_pair_details"][0]["reasons"])

    def test_missing_game_metadata_remains_an_exclusion(self):
        result=progress.pair_results([game(100,0,"guest_win")],100,1)
        self.assertEqual(result["eligible_pairs"],0)
        self.assertEqual(result["paired_wdl"]["games"],0)
        self.assertIn("missing-game-metadata",result["excluded_pair_details"][0]["reasons"])

    def test_broken_pair_identity_is_rejected(self):
        first,second=game(100,0),game(100,1)
        for change in ({"candidate_host":True},{"starting_flower":"W3"},{"host_seed":"other"}):
            with self.assertRaises(ValueError):progress.pair_results([first,{**second,**change}],100,1)
        with self.assertRaisesRegex(ValueError,"duplicate"):progress.pair_results([first,first],100,1)

    def test_length_turns_are_engine_counts_not_half_decisions(self):
        result=progress.lengths([game(100,0,decisions=28,turns=21),game(100,1,decisions=18,turns=13)])
        self.assertEqual(result["decisions"]["mean"],23)
        self.assertEqual(result["completed_turns"]["mean"],17)
        self.assertNotEqual(result["completed_turns"]["mean"],result["decisions"]["mean"]/2)

    def test_quicker_losses_do_not_hide_longer_wins(self):
        result=progress.lengths_by_result([game(100,0,"host_win",decisions=200,turns=160),
                                          game(100,1,"host_win",decisions=10,turns=8)])
        self.assertEqual(result["win"]["decisions"]["median"],200)
        self.assertEqual(result["loss"]["decisions"]["median"],10)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.psr=(Path(__file__).resolve().parents[1]/"benchmarks/results/human-mcts-bridge-2026-09-07/human.psr").read_text()

    def write(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))

    def run_fixture(self,reference=512,candidate=32,rules=None):
        root=self.root/"run"
        model={"feature_schema":"paisho-compact-value-features-v1","weights":[0.0]*64}
        self.write(root/"candidate.json",model)
        plan={"schema":"paisho-compact-comparison-plan-v1","candidate":{
            "snapshot":"candidate.json","sha256":progress.sha256((root/"candidate.json").read_bytes()),"training_steps":64},
            "reference":LEGACY_COMPARISON_REFERENCE,"build":{"source_sha256":"source"},
            "mcts":{"simulations":candidate},"reference_simulations":reference,"pairs":1,"first_pair":100,
            "decision_limit":2048,"workers":2}
        if rules is not None:
            plan["rules"]=rules
        self.write(root/"plan.json",plan)
        psr=self.psr if rules is None else self.psr.replace(progress.HISTORICAL_RULES,rules,1)
        for leg in (0,1):
            item=game(100,leg,"guest_win")
            item["record"]=f"records/game-{leg:08}.psr"
            item["record_sha256"]=progress.sha256(psr.encode())
            self.write(root/f"games/game-{leg:08}.json",item)
            (root/"records").mkdir(exist_ok=True)
            (root/item["record"]).write_text(psr)
        return root

    def learned_reference(self,root):
        plan=json.loads((root/"plan.json").read_text())
        self.write(root/"reference.json",{"feature_schema":"paisho-compact-value-features-v1","weights":[1.0]+[0.0]*63})
        plan.update(reference_kind="compact-model",reference="CompactValueModel from frozen reference.json; learned coefficients",
            reference_model={"snapshot":"reference.json","sha256":progress.sha256((root/"reference.json").read_bytes()),"training_steps":500})
        reference=progress.comparison_reference(plan,root)
        plan["evaluators"]=progress.comparison_evaluators(plan,reference)
        self.write(root/"plan.json",plan)
        for path in (root/"games").glob("*.json"):
            item=json.loads(path.read_text());item["evaluators"]=plan["evaluators"];self.write(path,item)
        return plan

    def anchor_fixture(self):
        path=self.root/"anchor.json"
        self.write(path,{"schema":"paisho-human-elo-anchor-v1","groups":[{
            "agent_label":"mcts-512","bot_cohort":"old","unique_games":6,"bot_binary_identity":None,
            "record_sha256":["old-human-1"],"baseline":{"posterior_mode":791,"credible_95":[291,1115]},
            "one_observation_per_group_sensitivity":{"posterior":{"posterior_mode":857,"credible_95":[331,1256]}}}]})
        return path

    def test_projection_reuses_anchor_without_creating_new_human_observations(self):
        root=self.run_fixture();anchor=self.anchor_fixture()
        result=progress.report([root],anchor)
        run=result["runs"][0]
        self.assertEqual(run["identity"]["rules"],progress.HISTORICAL_RULES)
        self.assertEqual(run["paired_results"]["paired_wdl"]["score"],0.5)
        self.assertEqual(run["score_equivalent_elo_delta"]["value"],0)
        projection=run["human_projection"]
        self.assertEqual(projection["new_human_games_for_this_model"],0)
        self.assertEqual(projection["historical_human_games"],6)
        self.assertEqual(projection["scenarios"][0]["working_site_elo"],791)
        self.assertEqual(projection["scenarios"][2]["working_site_elo"],857)
        self.assertIn("NOT a calibrated joint",projection["scenarios"][0]["bounds_label"])
        self.assertIn("ancien MCTS-512",progress.markdown(result))

    def test_corrected_rules_keep_scores_without_inheriting_historical_human_elo(self):
        rules=progress.HISTORICAL_RULES+"-v2"
        root=self.run_fixture(rules=rules)
        run=progress.report([root],self.anchor_fixture())["runs"][0]
        self.assertEqual(run["identity"]["rules"],rules)
        self.assertEqual(run["paired_results"]["paired_wdl"]["score"],0.5)
        self.assertEqual(run["human_projection"]["status"],"no-cross-rules-human-bridge")
        self.assertNotIn("scenarios",run["human_projection"])
        plan=json.loads((root/"plan.json").read_text())
        del plan["rules"]
        self.write(root/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"PSR rules disagree"):
            progress.read_run(root)

    def test_non512_reference_cannot_be_projected_onto512_anchor(self):
        root=self.run_fixture(reference=32)
        result=progress.report([root],self.anchor_fixture())
        self.assertEqual(result["runs"][0]["human_projection"]["status"],"no-direct-mcts512-bridge")

    def test_learned512_keeps_its_exact_parent_identity_without_old_human_transfer(self):
        root=self.run_fixture();plan=self.learned_reference(root)
        result=progress.report([root],self.anchor_fixture());run=result["runs"][0]
        self.assertEqual(run["identity"]["reference_kind"],"compact-model")
        self.assertEqual(run["identity"]["reference_model_sha256"],plan["reference_model"]["sha256"])
        self.assertEqual(run["identity"]["reference_training_steps"],500)
        self.assertEqual(run["human_projection"]["status"],"no-direct-legacy-human-bridge")
        self.assertEqual(run["files_sha256"]["reference.json"],plan["reference_model"]["sha256"])
        self.assertIn("contre modèle figé MCTS-512",progress.markdown(result))

    def test_retained_heuristic_keeps_scores_but_cannot_inherit_old_human_anchor(self):
        root=self.run_fixture();plan=json.loads((root/"plan.json").read_text())
        plan.update(reference_kind="legacy-cpu-heuristic-retained",reference_reuse=True,candidate_reuse=False)
        reference=progress.comparison_reference(plan,root)
        plan["evaluators"]=progress.comparison_evaluators(plan,reference)
        self.write(root/"plan.json",plan)
        for path in (root/"games").glob("*.json"):
            item=json.loads(path.read_text());item["evaluators"]=plan["evaluators"];self.write(path,item)
        result=progress.report([root],self.anchor_fixture());run=result["runs"][0]
        self.assertEqual(run["paired_results"]["paired_wdl"]["score"],0.5)
        self.assertEqual(run["human_projection"]["status"],"no-direct-legacy-human-bridge")
        self.assertTrue(run["identity"]["reference_reuse"])
        self.assertIn("heuristique avec rétention",progress.markdown(result))
        item=json.loads((root/"games/game-00000000.json").read_text())
        item["evaluators"]["reference_reuse"]=False;self.write(root/"games/game-00000000.json",item)
        with self.assertRaisesRegex(ValueError,"game evaluator identities"):progress.read_run(root)

    def test_legacy_label_cannot_hide_retained_search_or_nonboolean_flags(self):
        root=self.run_fixture();plan=json.loads((root/"plan.json").read_text())
        plan.update(reference_reuse=True);self.write(root/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"legacy comparison reference identity mismatch"):progress.read_run(root)
        plan.update(reference_reuse="false");self.write(root/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"must be booleans"):progress.read_run(root)

    def test_unknown_reference_cannot_inherit_human_anchor_by_budget_alone(self):
        root=self.run_fixture();plan=json.loads((root/"plan.json").read_text())
        plan["reference"]="some 512 opponent";self.write(root/"plan.json",plan)
        run=progress.report([root],self.anchor_fixture())["runs"][0]
        self.assertEqual(run["identity"]["reference_kind"],"unknown")
        self.assertEqual(run["human_projection"]["status"],"no-direct-legacy-human-bridge")

    def test_reference_snapshot_or_game_role_hash_mismatch_is_rejected(self):
        root=self.run_fixture();self.learned_reference(root)
        path=root/"games/game-00000001.json";item=json.loads(path.read_text())
        original=dict(item["evaluators"])
        item["evaluators"]["reference_artifact_sha256"]="another-parent";self.write(path,item)
        with self.assertRaisesRegex(ValueError,"game evaluator identities"):progress.read_run(root)
        item["evaluators"]=original;self.write(path,item)
        (root/"reference.json").write_text("changed")
        with self.assertRaisesRegex(ValueError,"reference snapshot hash mismatch"):progress.read_run(root)

    def test_legacy_label_cannot_hide_a_learned_artifact(self):
        root=self.run_fixture();plan=self.learned_reference(root)
        plan.update(reference_kind="legacy-cpu-heuristic",reference=LEGACY_COMPARISON_REFERENCE)
        self.write(root/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"legacy comparison reference identity mismatch"):progress.read_run(root)

    def test_incomplete_archive_cannot_be_projected(self):
        root=self.run_fixture();(root/"games/game-00000001.json").unlink()
        result=progress.report([root],self.anchor_fixture())["runs"][0]
        self.assertFalse(result["complete_archive"])
        self.assertEqual(result["human_projection"]["status"],"withheld-incomplete-or-failed-archive")

    def test_outcome_and_turns_are_checked_when_independent_replay_supplied(self):
        root=self.run_fixture()
        facts={"outcome":"guest","decisions":28,"completed_turns":21}
        with patch.object(progress,"replay_record",return_value=(self.psr.encode(),facts)) as replay:
            result=progress.read_run(root,Path("unused-verifier"))
            self.assertEqual(replay.call_count,2)
            self.assertEqual(result["paired_results"]["eligible_lengths"]["completed_turns"]["median"],21)
        with patch.object(progress,"replay_record",return_value=(self.psr.encode(),{**facts,"completed_turns":14})):
            with self.assertRaisesRegex(ValueError,"completed turns"):progress.read_run(root,Path("unused-verifier"))

    def test_corrupt_psr_or_summary_is_rejected(self):
        root=self.run_fixture()
        self.write(root/"summary.json",{"schema":"paisho-compact-comparison-summary-v1","status":"completed",
            "paired_comparison":{"wins":1,"draws":0,"losses":1,"rated_pairs":1,"excluded":0,"score":0.9}})
        with self.assertRaisesRegex(ValueError,"summary disagrees"):progress.read_run(root)
        (root/"summary.json").unlink()
        (root/"records/game-00000000.psr").write_text("altered")
        with self.assertRaisesRegex(ValueError,"PSR hash mismatch"):progress.read_run(root)

    def test_extreme_result_keeps_raw_infinity_separate_from_finite_prior_estimate(self):
        root=self.run_fixture()
        path=root/"games/game-00000000.json";item=json.loads(path.read_text())
        item["termination"]["outcome"]="host_win";self.write(path,item)
        run=progress.report([root],self.anchor_fixture())["runs"][0]
        self.assertEqual(run["score_equivalent_elo_delta"]["boundary"],"positive-infinity")
        self.assertTrue(math.isfinite(run["regularized_pair_delta"][0]["posterior_mode"]))
        self.assertIsNone(run["human_projection"]["scenarios"][0]["site_elo_using_raw_score_delta"])

    def test_report_sorts_candidate_budgets_numerically(self):
        paths=[self.root/"a256",self.root/"z32"]
        def fake_run(path,verifier):
            return {"run":str(path),"identity":{"candidate_budget":256 if path.name=="a256" else 32,
                                                   "reference_budget":512}}
        with patch.object(progress,"read_run",side_effect=fake_run):
            result=progress.report(paths)
        self.assertEqual([run["identity"]["candidate_budget"] for run in result["runs"]],[32,256])


if __name__=="__main__":
    unittest.main()
