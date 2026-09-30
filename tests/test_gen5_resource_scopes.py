"""Lightweight use must not silently require the historical campaign store."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import assets

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gen5_manage', ROOT/'experimental/gen5/manage.py')
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)


class Gen5ResourceScopes(unittest.TestCase):
    def test_default_profiles_exclude_historical_fifo_and_full_store(self):
        for profile in ['gen5-play', 'gen5']:
            names = {n for g in assets.PROFILES[profile] for n in assets.entries_for(ROOT, g)}
            self.assertFalse(any(n.startswith(('assets/gen5-history/', 'assets/gen5-replay/files/')) for n in names))
            self.assertNotIn('assets/gen5-replay/replay.index.json', names)
        self.assertIn('gen5-history', assets.PROFILES['gen5-history'])
        catalog = json.loads((ROOT/'data/release-assets.json').read_text())
        self.assertEqual(catalog['profiles'], assets.PROFILES)

    def test_play_preparation_with_only_play_assets_and_no_human_or_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root/'experimental/gen5'
            (exp/'models').mkdir(parents=True)
            (exp/'inputs').mkdir()
            manifest = root/'memory/test/summary.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}')
            (exp/'models/accepted.json').write_text(json.dumps({'memory_manifest':'memory/test/summary.json'}))
            (exp/'inputs/reference.json').write_text('{}')
            # Deliberately no human dataset, replay index, FIFO or historical archives.
            with patch.object(manage,'ROOT',root), patch.object(manage,'EXP',exp), patch.object(manage,'require_groups') as required:
                output = manage.prepare(training=False)
            required.assert_called_once_with(['core-memory','gen5-memory'])
            model = json.loads((output/'models/accepted.json').read_text())
            self.assertEqual(model['memory_manifest'],str(manifest))
            self.assertEqual(model['memory_manifest_sha256'],manage.sha(manifest))


if __name__ == '__main__':
    unittest.main()
