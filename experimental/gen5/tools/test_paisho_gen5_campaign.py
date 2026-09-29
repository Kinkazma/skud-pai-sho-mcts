import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('gen5_campaign', Path(__file__).with_name('paisho_gen5_campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def inputs(self, root):
        binary = root / 'native'
        binary.write_text('#!/bin/sh\nexit 0\n')
        binary.chmod(0o755)
        model = root / 'model.json'
        model.write_text('{}')
        config = root / 'config.json'
        config.write_text(json.dumps({'model': str(model), 'budgets': [[256, .6], [2048, .4]]}))
        end = dt.datetime.fromtimestamp(time.time()+14*3600, dt.timezone.utc).isoformat()
        return binary, config, end

    def test_frozen_inputs_absolute_deadline_and_duplicate_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, config, end = self.inputs(root)
            out = root / 'run'
            options=json.loads(config.read_text())
            options['historical_unlimited_budgets'] = [32]
            options['historical_seconds'] = [[32, 3.25], [64, 7.75], [128, .75]]
            for key in ['evaluation_anchor','resume_progress','publication_validation']:
                path=root/(key+'.json');path.write_text('{}');options[key]=str(path)
            config.write_text(json.dumps(options))
            with patch.object(campaign.subprocess, 'check_output', return_value='test-revision'), patch.object(campaign.subprocess, 'Popen', return_value=Mock(pid=123)):
                campaign.start(config, binary, out, end)
            frozen = json.loads((out / 'config.json').read_text())
            self.assertEqual(frozen['end_unix_seconds'], dt.datetime.fromisoformat(end).timestamp())
            self.assertTrue(frozen['learn'])
            self.assertEqual(frozen['seconds'], 86400)
            self.assertEqual(frozen['historical_unlimited_budgets'], [32, 64, 128])
            self.assertEqual(json.loads(config.read_text())['historical_unlimited_budgets'], [32])
            self.assertEqual(Path(frozen['evaluation_anchor']).name,'evaluation-anchor.json')
            self.assertEqual(Path(frozen['resume_progress']).name,'resume-progress.json')
            self.assertEqual(Path(frozen['publication_validation']).name,'publication-validation.json')
            self.assertIn('publication-validation.json', json.loads((out / 'plan.json').read_text())['hashes'])
            self.assertFalse(json.loads((out / 'plan.json').read_text())['automatic_resume'])
            self.assertEqual(campaign.execute(out), 0)
            with self.assertRaises(FileExistsError):
                campaign.execute(out)
            (out / 'initial-model.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                campaign.execute(out)

    def test_invalid_budgets_and_past_deadline_do_not_create_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, config, end = self.inputs(root)
            out = root / 'run'
            config.write_text(json.dumps({'budgets': [[2049, 1.0]]}))
            with self.assertRaisesRegex(ValueError, 'at most 2048'):
                campaign.start(config, binary, out, end)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, 'next twenty-four hours'):
                campaign.start(config, binary, out, '2000-01-01T23:00:00+01:00')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
