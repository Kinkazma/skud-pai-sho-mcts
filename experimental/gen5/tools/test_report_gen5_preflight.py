import unittest
from report_gen5_preflight import curve


def game(seconds, terminal=True, cap=30, censored=False):
    return {'seconds':seconds,'termination':'rules-terminal' if terminal else 'wall-limit',
            'error':None,'campaign_censored':censored,'cap_seconds':cap,'eligible_examples':100}


class CutoffTests(unittest.TestCase):
    def test_no_per_game_deadline_is_readable_and_campaign_end_remains_censored(self):
        point=curve([game(4,cap=None),game(7,False,cap=None,censored=True)],6)
        self.assertEqual(point['terminal_games'],1)
        self.assertEqual(point['occupied_worker_seconds'],4)

    def test_timeout_cost_is_included_and_no_completion_quota_is_imposed(self):
        cohort=[game(2),game(4),game(20)]
        point=curve(cohort,4)
        self.assertEqual(point['terminal_games'],2)
        self.assertEqual(point['occupied_worker_seconds'],10)
        self.assertAlmostEqual(point['terminal_games_per_worker_second'],.2)
        self.assertGreater(point['terminal_games_per_worker_second'],curve(cohort,20)['terminal_games_per_worker_second'])

    def test_own_timeout_remains_in_denominator_but_campaign_drain_is_excluded(self):
        point=curve([game(2),game(3,False,cap=3),game(.1,False,censored=True)],3)
        self.assertEqual(point['attempts'],2)
        self.assertEqual(point['occupied_worker_seconds'],5)
        self.assertEqual(point['terminal_games'],1)
        with self.assertRaisesRegex(ValueError,'censored observation'):
            curve([game(3,False,cap=3)],4)

    def test_cycle_examples_do_not_count_as_terminal_training_positions(self):
        cycle=game(3,False);cycle['termination']='repetition-training-loss'
        point=curve([game(2),cycle],5)
        self.assertEqual(point['terminal_positions'],100)
        self.assertEqual(point['occupied_worker_seconds'],5)


if __name__=='__main__':unittest.main()
