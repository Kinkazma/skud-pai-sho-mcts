import unittest
from benchmark_mcts_deadline_sweep import summarize


class ThroughputTest(unittest.TestCase):
    def test_timeouts_cost_time_but_produce_no_games(self):
        result = summarize([
            {'status':'accepted','elapsed_seconds':2,'decisions':40,'psr_sha256':'a'},
            {'status':'accepted','elapsed_seconds':4,'decisions':80,'psr_sha256':'b'},
            {'status':'timeout','elapsed_seconds':6},
        ], workers=2, wall=8)
        self.assertEqual(result['accepted'],2)
        self.assertEqual(result['saturated_games_per_second'],1/3)
        self.assertEqual(result['observed_games_per_second'],0.25)
        self.assertEqual(result['estimated_minutes_per_10000'],500)
        self.assertEqual(result['unique_psrs'],2)

    def test_zero_yield_has_no_finite_eta(self):
        result = summarize([{'status':'timeout','elapsed_seconds':1}],10,1)
        self.assertEqual(result['saturated_games_per_second'],0)
        self.assertIsNone(result['estimated_minutes_per_10000'])


if __name__=='__main__':
    unittest.main()
