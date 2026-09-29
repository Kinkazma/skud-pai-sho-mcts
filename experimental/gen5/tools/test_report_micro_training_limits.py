import json
from pathlib import Path
import tempfile
import unittest

from report_micro_training_limits import curve, read_training_rows


def row(i, seconds, decisions, kind='rules-terminal'):
    return {'id': i, 'seconds': seconds, 'decisions': decisions,
            'termination': kind, 'error': None, 'targets_file': f'{i}.gz'}


class TrainingLimitsTest(unittest.TestCase):
    def test_interleaved_evaluation_ids_do_not_duplicate_training(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'log'
            records = [row(0, 1, 20), {'id': 0, 'termination': 'rules-terminal'}, row(1, 2, 40)]
            p.write_text('\n'.join(json.dumps(r) for r in records))
            rows, excluded = read_training_rows(p)
            self.assertEqual([r['id'] for r in rows], [0, 1])
            self.assertEqual(excluded, 1)
            p.write_text(json.dumps(records[0])+'\n'+json.dumps(records[0]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_training_rows(p)

    def test_timeouts_cost_compute_and_cycles_do_not_count_as_terminal(self):
        rows = [row(0, 1, 10), row(1, 3, 50), row(2, 4, 200, 'repetition-training-loss')]
        at_two = curve(rows, 2)
        self.assertEqual(at_two['occupied_worker_seconds'], 5)
        self.assertEqual(at_two['terminal_games'], 1)
        self.assertEqual(at_two['terminal_positions'], 10)
        self.assertEqual(curve(rows, 4)['terminal_games'], 2)

    def test_final_drain_excluded_and_games_positions_have_different_optima(self):
        rows = [row(0, 1, 10), row(1, 1, 10), row(2, 3, 100), row(3, .1, 1, 'wall-limit')]
        early, late = curve(rows, 1), curve(rows, 3)
        self.assertEqual(early['cohort_attempts'], 3)
        self.assertEqual(early['occupied_worker_seconds'], 3)
        self.assertGreater(early['terminal_games_per_worker_second'], late['terminal_games_per_worker_second'])
        self.assertGreater(late['terminal_positions_per_worker_second'], early['terminal_positions_per_worker_second'])


if __name__ == '__main__':
    unittest.main()
