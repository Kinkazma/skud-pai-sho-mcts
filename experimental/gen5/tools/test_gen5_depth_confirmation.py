import itertools
import unittest
from gen5_depth_confirmation import confidence_interval, log_capital, pair_difference
import math


class ConfirmationStatisticsTests(unittest.TestCase):
    def test_complete_census_exact(self):
        self.assertEqual(confidence_interval([-1, 0, .5, 1], 4), (.125, .125))

    def test_without_replacement_conditional_martingale(self):
        # Enumerated population: conditional expected capital multiplier is 1.
        remaining = [-1, -.5, 0, .5, 1]
        conditional = sum(remaining)/len(remaining)
        for stake in [-.49, -.1, .1, .49]:
            self.assertAlmostEqual(sum(1+stake*(x-conditional) for x in remaining)/len(remaining), 1.)

    def test_all_permutations_optional_stopping_coverage(self):
        population = [-1, -.5, 0, 0, .5, 1]
        mean = sum(population)/len(population)
        orders = set(itertools.permutations(population))
        failed = 0
        for values in orders:
            crossed = False
            for n in range(1, len(values)):
                if any(log_capital(values[:n], len(values), mean, sign) >= math.log(1/.1)
                       for sign in (-1, 1)):
                    crossed = True
            failed += crossed
        self.assertLessEqual(failed/len(orders), .2)

    def test_paired_seats_unresolved_not_draw(self):
        games = [dict(start_index=7,budget=32,floor=f,seat=s,outcome=o) for f,s,o in
                 [(0,'H','win'),(0,'G','loss'),(16,'H','unresolved'),(16,'G','win')]]
        self.assertEqual(pair_difference(games,7,32,16), 0.)
        with self.assertRaises(ValueError):
            pair_difference(games[:-1],7,32,16)

    def test_bounds_narrow_and_classify_known_population(self):
        lo,hi=confidence_interval([0.] * 600, 1000)
        self.assertLess(hi,.05)
        self.assertGreater(lo,-.05)
        lo,hi=confidence_interval([.5] * 300, 1000)
        self.assertGreater(lo,.05)


if __name__ == '__main__':
    unittest.main()
