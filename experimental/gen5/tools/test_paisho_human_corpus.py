#!/usr/bin/env python3
"""Regression checks for immutable human-game imports and provisional calibration."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import paisho_human_corpus as corpus

VERIFIER = Path(os.environ.get("PAISHO_RECORD_VERIFIER", corpus.ROOT / "target/release/examples/verify_record"))
FIXTURE = corpus.ROOT / "benchmarks/results/human-mcts-bridge-2026-09-07/human.psr"


def observation(score, rating=1093):
    return {"metadata": {"human_rating": rating}, "bot_score": score}


class PosteriorTests(unittest.TestCase):
    def test_prior_only_stays_centered(self):
        result = corpus.posterior([])
        self.assertEqual(result["posterior_mode"], 1000)
        self.assertAlmostEqual(result["posterior_mean"], 1000, places=8)

    def test_reproduces_previous_one_loss_anchor(self):
        result = corpus.posterior([observation(0)])
        self.assertEqual(result["posterior_mode"], 882)
        self.assertEqual(result["credible_95"], [347, 1334])
        self.assertAlmostEqual(result["posterior_mean"], 858.5039688766783, places=8)

    def test_win_loss_symmetry_and_draw(self):
        win = corpus.posterior([observation(1, 1000)])
        loss = corpus.posterior([observation(0, 1000)])
        draw = corpus.posterior([observation(0.5, 1000)])
        self.assertAlmostEqual(win["posterior_mean"] + loss["posterior_mean"], 2000, places=8)
        self.assertEqual(draw["posterior_mode"], 1000)
        self.assertGreater(win["posterior_mode"], 1000)
        self.assertLess(loss["posterior_mode"], 1000)

    def test_dependence_weight_reduces_repeated_evidence(self):
        one = corpus.posterior([observation(0)])
        reduced = corpus.posterior([observation(0)] * 6, evidence_weight=1 / 6)
        six = corpus.posterior([observation(0)] * 6)
        self.assertAlmostEqual(one["posterior_mean"], reduced["posterior_mean"], places=8)
        self.assertLess(six["posterior_mode"], one["posterior_mode"])

    def test_metadata_rejects_nonfinite_rating_and_bad_side(self):
        for changes in ({"elo": True}, {"elo": float("nan")}, {"human_side": "black"}):
            data = {"agent": "mcts-512", "elo": 1093, "human_side": "guest", **changes}
            with self.assertRaises(ValueError):
                corpus.metadata(data)

    def test_per_group_weight_preserves_two_independent_players(self):
        two = [observation(0), observation(0, 1300)]
        six = [observation(0)] + [{**observation(0, 1300), "observation_weight": 0.2}] * 5
        self.assertAlmostEqual(corpus.posterior(two)["posterior_mean"],
                               corpus.posterior(six)["posterior_mean"], places=8)


@unittest.skipUnless(VERIFIER.is_file(), "build paisho-core --example verify_record first")
class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "exports"
        self.source.mkdir()
        self.output = self.root / "corpus"
        self.psr = FIXTURE.read_text()

    def export(self, name, **overrides):
        data = {"agent": "mcts-512", "elo": 1093, "human_side": "guest", "moves": self.psr}
        data.update(overrides)
        path = self.source / name
        path.write_text(json.dumps(data))
        return path

    def run_import(self, output=None, cohort="legacy-test"):
        return corpus.import_corpus(self.source, output or self.output, VERIFIER, cohort)

    def test_exact_duplicate_preserves_aliases_and_is_counted_once(self):
        first = self.export("a.json")
        self.export("a-copy.json")
        manifest = self.run_import()
        self.assertEqual(manifest["counts"], {
            "source_files": 2, "unique_original_bytes": 1, "unique_replayed_games": 1,
            "calibration_eligible_games": 1, "rejected_source_files": 0})
        self.assertEqual(len(manifest["records"][0]["source_indexes"]), 2)
        stored = self.output / manifest["sources"][0]["stored_path"]
        self.assertEqual(stored.read_bytes(), first.read_bytes())
        result = corpus.calibration([self.output / "manifest.json"])
        self.assertEqual(result["groups"][0]["bot_wins_draws_losses"], [0, 0, 1])
        self.assertEqual(result["groups"][0]["explicit_human_ids"], 0)

    def test_whitespace_duplicate_gets_one_canonical_record(self):
        self.export("a.json")
        self.export("b.json", moves=self.psr.replace("\n", "\r\n") + "\n")
        manifest = self.run_import()
        self.assertEqual(manifest["counts"]["unique_original_bytes"], 2)
        self.assertEqual(manifest["counts"]["unique_replayed_games"], 1)

    def test_bot_score_is_from_human_side_and_replayed_outcome(self):
        self.export("a.json", human_side="host")
        game = self.run_import()["records"][0]
        self.assertEqual(game["replay"]["outcome"], "guest")
        self.assertEqual(game["bot_score"], 1)

    def test_conflicting_duplicate_metadata_is_not_rated(self):
        self.export("a.json")
        self.export("b.json", elo=1300)
        manifest = self.run_import()
        self.assertFalse(manifest["records"][0]["calibration_eligible"])
        self.assertEqual(corpus.calibration([self.output / "manifest.json"])["unique_eligible_games"], 0)

    def test_illegal_and_unfinished_records_are_preserved_not_rated(self):
        self.export("illegal.json", moves=self.psr + "plant R3 0,0\n")
        self.export("ongoing.json", moves=self.psr.split("actions\n")[0] + "actions\n")
        self.export("invalid-side.json", human_side="black")
        manifest = self.run_import()
        self.assertEqual(manifest["counts"]["source_files"], 3)
        self.assertEqual(manifest["counts"]["rejected_source_files"], 2)
        self.assertEqual(manifest["counts"]["calibration_eligible_games"], 0)
        self.assertEqual(len(list((self.output / "originals").glob("*.json"))), 3)

    def test_repeated_import_is_deduplicated_and_cohort_conflict_fails(self):
        self.export("a.json")
        self.run_import()
        other = self.root / "other"
        self.run_import(other)
        manifests = [self.output / "manifest.json", other / "manifest.json"]
        self.assertEqual(corpus.calibration(manifests)["unique_eligible_games"], 1)
        different = self.root / "different"
        self.run_import(different, cohort="other-version")
        with self.assertRaisesRegex(ValueError, "conflicting cohort"):
            corpus.calibration([manifests[0], different / "manifest.json"])

    def test_conflict_in_one_import_cannot_be_ignored_by_another(self):
        self.export("a.json")
        self.run_import()
        self.export("b.json", elo=1300)
        other = self.root / "other"
        self.run_import(other)
        with self.assertRaisesRegex(ValueError, "eligible in one import"):
            corpus.calibration([self.output / "manifest.json", other / "manifest.json"])

    def test_import_never_overwrites_and_calibration_verifies_originals(self):
        self.export("a.json")
        manifest = self.run_import()
        with self.assertRaises(FileExistsError):
            self.run_import()
        original = self.output / manifest["sources"][0]["stored_path"]
        original.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original hash mismatch"):
            corpus.calibration([self.output / "manifest.json"])

    def test_explicit_player_group_requires_complete_provenance_mapping(self):
        self.export("a.json")
        manifest = self.run_import()
        mapping_file = self.root / "player-groups.json"
        mapping = {"schema": "paisho-human-player-groups-v1", "source": "test declaration",
                   "record_groups": {manifest["records"][0]["record_sha256"]: "player-a"}}
        mapping_file.write_text(json.dumps(mapping))
        result = corpus.calibration([self.output / "manifest.json"], mapping_file)
        self.assertEqual(result["groups"][0]["declared_player_groups"], {"player-a": 1})
        mapping["record_groups"] = {}
        mapping_file.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, "explicitly cover"):
            corpus.calibration([self.output / "manifest.json"], mapping_file)


if __name__ == "__main__":
    unittest.main()
