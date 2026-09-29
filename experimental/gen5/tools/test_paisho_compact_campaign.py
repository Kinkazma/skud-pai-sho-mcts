import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import paisho_compact_campaign as campaign


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "fake-engine"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self.model = self.root / "model.json"
        self.model.write_text("{}")
        self.run = self.root / "campaign"

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_inputs_and_duplicate_run(self):
        plan = campaign.prepare(self.run, self.binary, self.model, 60)
        self.binary.write_text("#!/bin/sh\nexit 42\n")
        self.model.write_text("changed")
        self.assertEqual(campaign.execute(self.run), 0)
        self.assertEqual(json.loads((self.run / "status.json").read_text())["status"], "completed")
        self.assertEqual(plan["start_positions"], "standard-only")
        with self.assertRaises(FileExistsError):
            campaign.execute(self.run)
        with self.assertRaises(FileExistsError):
            campaign.prepare(self.run, self.binary, self.model)

    def test_changed_frozen_binary_never_runs(self):
        campaign.prepare(self.run, self.binary, self.model)
        frozen = self.run / "bin/paisho-compact"
        frozen.write_text("#!/bin/sh\nexit 0\n# changed\n")
        self.assertEqual(campaign.execute(self.run), 1)
        result = json.loads((self.run / "status.json").read_text())
        self.assertIn("frozen input changed", result["error"])
        self.assertNotIn("trainer_pid", result)

    def test_historical_gen3_preserves_actual_original_learning_recipe(self):
        plan = campaign.prepare(self.run, self.binary, self.model,
                                learning_profile="historical-gen3")
        flags = dict(zip(plan["command"][2::2], plan["command"][3::2]))
        self.assertEqual(flags["--replay-capacity"], "0")
        self.assertEqual(flags["--replay-ratio"], "0")
        self.assertEqual(flags["--repetition-cycles"], "0")
        self.assertEqual(flags["--reuse-search"], "false")
        self.assertEqual(flags["--budgets"], "512,256,128,64,32")
        self.assertEqual(flags["--learning-rate"], "0.01")
        self.assertEqual(flags["--lambda"], "0.5")
        self.assertEqual(flags["--samples"], "128")
        self.assertEqual(flags["--seconds"], "3600")
        self.assertEqual(plan["learning_profile"], "historical-gen3")

    def test_failed_status_publication_does_not_orphan_child(self):
        campaign.prepare(self.run, self.binary, self.model)
        child = mock.Mock()
        child.pid = 12345
        child.poll.return_value = None
        original = campaign.atomic_json
        def failed_first_publication(path, data):
            if data["status"] == "training":
                raise OSError("status write failed")
            original(path, data)
        with mock.patch.object(campaign.subprocess, "Popen", return_value=child), \
                mock.patch.object(campaign, "atomic_json", side_effect=failed_first_publication):
            self.assertEqual(campaign.execute(self.run), 1)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
