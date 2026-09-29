import unittest
from gen5_depth_curve import classify, paired, validate_population

class CurveTests(unittest.TestCase):
    def test_small_positive_is_not_rejected_by_five_point_threshold(self):
        self.assertEqual(classify(.005,.035,256,1479,128),'positive_precise')
        self.assertIsNone(classify(.005,.09,256,1479,128))
        self.assertEqual(classify(-.01,.015,1024,1479,128),'bounded_within_two_percentage_points')
    def test_census_and_minimum(self):
        self.assertEqual(classify(.001,.001,1479,1479,128),'positive_census')
        self.assertIsNone(classify(.01,.02,32,1479,128))
    def test_five_reference_and_exact_cache_population(self):
        g=[dict(start_index=4,budget=64,floor=f,seat=s,outcome=o) for f,s,o in
           [(5,'H','win'),(5,'G','loss'),(8,'H','win'),(8,'G','win')]]
        self.assertEqual(paired(g,4,64,8,5),.5)
        validate_population({'seed':1},{'seed':1})
        with self.assertRaises(RuntimeError):validate_population({'seed':1},{'seed':2})

if __name__=='__main__':unittest.main()
