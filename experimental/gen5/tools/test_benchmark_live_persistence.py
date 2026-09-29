import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_live_persistence import all_shards_match, stopped_group


class PauseEvidenceTests(unittest.TestCase):
    def test_matched_claim_requires_every_shard_of_every_cycle(self):
        def run(*digests):
            return {"cycles": [{"replay_sha256": digest} for digest in digests]}
        self.assertTrue(all_shards_match([run("a", "b")] * 4, 2))
        self.assertFalse(all_shards_match([run("a", "b"), run("a", "c")], 2))
        self.assertFalse(all_shards_match([run("a"), run("a")], 2))
    def check(self, rows):
        config = SimpleNamespace(state_path=Path("unused"))
        with patch.object(Path, "read_text", return_value=json.dumps({"process_pid": 100})), \
                patch("benchmark_live_persistence.subprocess.check_output", return_value=rows):
            return stopped_group(config)

    def test_checks_every_group_member_not_only_leader(self):
        with self.assertRaises(RuntimeError):
            self.check("100 100 T\n101 100 S\n200 200 R\n")

    def test_accepts_stopped_group_and_ignores_unrelated_process(self):
        rows = self.check("100 100 T\n101 100 T+\n200 200 R\n")
        self.assertEqual([row["pid"] for row in rows], [100, 101])

    def test_missing_leader_is_not_evidence(self):
        with self.assertRaises(RuntimeError):
            self.check("101 100 T\n")


if __name__ == "__main__":
    unittest.main()
