import json
import tempfile
import unittest
from pathlib import Path
from gen5_r1_freeze import audit_groups


class LeakageTests(unittest.TestCase):
    def test_transitive_cross_split_cluster_is_quarantined(self):
        rows = [('a', False, 'x'), ('b', False, 'x'), ('b', False, 'y'),
                ('c', True, 'y'), ('clean', True, 'z')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'index.jsonl'
            path.write_text(''.join(json.dumps({'group': g, 'held_out': h,
                                                'exact': f'{g}-{x}', 'symmetry': x}) + '\n'
                                    for g, h, x in rows))
            result = audit_groups([path])
        self.assertEqual(result['quarantined_groups'], ['a', 'b', 'c'])
        self.assertEqual(result['cross_split_symmetric_positions'], 1)
        self.assertEqual(result['cross_split_exact_positions'], 0)

    def test_source_split_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'index.jsonl'
            path.write_text('\n'.join(json.dumps({'group': 'a', 'held_out': h,
                                                 'exact': 'x', 'symmetry': 'x'})
                                      for h in [False, True]))
            with self.assertRaises(ValueError):
                audit_groups([path])


if __name__ == '__main__':
    unittest.main()
