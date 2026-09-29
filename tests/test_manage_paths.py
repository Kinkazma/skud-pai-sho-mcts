import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import manage


class LauncherOutputs(unittest.TestCase):
    def test_distinct_generation_names_keep_distinct_config_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'configs/gen3.5-historical.json').write_text('{"historical_pool": []}')
            models = root/'portable-models'
            calls = []
            for generation in ['3.2', '3.3']:
                args = ['manage.py', 'train', '--generation', generation, '--budget', '8',
                        '--heuristic-reference', '--output', 'runs/gen'+generation]
                with patch.object(manage, 'ROOT', root), patch.object(manage, 'prepare', return_value=models), \
                     patch.object(manage, 'run', side_effect=lambda a: calls.append(a)), patch.object(sys, 'argv', args):
                    manage.main()
            self.assertEqual(len(calls), 2)
            for generation in ['3.2', '3.3']:
                path = root/('runs/gen'+generation+'.config.json')
                config = json.loads(path.read_text())
                self.assertEqual(config['output'], str(root/('runs/gen'+generation)))
                self.assertTrue(config['model'].endswith('gen'+generation.replace('.', '-')+'.json'))
            self.assertFalse((root/'runs/gen3.config.json').exists())


if __name__ == '__main__':
    unittest.main()
