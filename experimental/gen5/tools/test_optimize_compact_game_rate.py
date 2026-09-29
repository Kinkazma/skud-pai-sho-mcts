import unittest
from analyze_compact_deadlines import Observation
from optimize_compact_game_rate import optimize


def game(seconds,terminal=True,roots=40):
    return Observation('fixture',seconds,'terminal' if terminal else 'time-censored',
                       100,roots,32,2048,'terminal' if terminal else 'game-deadline',
                       'host' if terminal else None)


class GameRateTests(unittest.TestCase):
    def test_restart_can_beat_waiting_for_every_game(self):
        result=optimize([game(20),game(60)],60)
        self.assertEqual(result['best']['seconds'],20)
        self.assertEqual(result['best']['completion_fraction'],0.5)
        self.assertAlmostEqual(result['best']['rate'],1/40)

    def test_no_percentage_or_position_volume_constraint(self):
        result=optimize([game(2,roots=1),game(30,roots=1000)],30)
        self.assertEqual(result['best']['seconds'],2)

    def test_rejections_cost_time_and_censoring_never_becomes_a_finish(self):
        result=optimize([game(2),game(3,False)],10)
        self.assertEqual(result['best']['seconds'],2)
        self.assertAlmostEqual(result['best']['rate'],0.25)
        self.assertTrue(all(r['rate'] is None for r in result['table'] if r['seconds']>3))

    def test_global_peak_not_first_flat_slope(self):
        result=optimize([game(1),game(4),game(4),game(4)],5)
        self.assertEqual(result['best']['seconds'],4)

    def test_no_finished_games_has_no_false_optimum(self):
        self.assertIsNone(optimize([game(30,False)],30)['best'])

    def test_equal_rate_chooses_shorter_cutoff(self):
        self.assertEqual(optimize([game(2)],30)['best']['seconds'],2)

    def test_missing_records_are_costed_without_inventing_outcomes(self):
        missing=Observation('missing',0,'time-censored',0,0,0,2048,'missing-record',None)
        result=optimize([game(2),game(4),missing],10,conservative=True)
        best=result['best']
        self.assertEqual(best['seconds'],4)
        self.assertIsNone(best['rate'])
        self.assertEqual(best['finished'],2)
        self.assertEqual(best['attempts'],3)
        self.assertAlmostEqual(best['selection_rate'],2/(2+4+4))
        self.assertEqual(best['rate_bounds'],[0.2,0.5])


if __name__=='__main__': unittest.main()
