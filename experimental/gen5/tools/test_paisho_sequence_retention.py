import unittest
from paisho_sequence_retention import Game,Policy,select
class RetentionTests(unittest.TestCase):
    def test_full_bank_replaces_and_never_grows(self):
        rows=[Game('human',True,('h',))]
        for i in range(100):
            rows,report=select(rows,[Game(str(i),motifs=(str(i),),last_added=i)],policy=Policy(capacity=8))
            self.assertLessEqual(len(rows),8)
            self.assertIn('human',{g.identity for g in rows})
        self.assertEqual(len(rows),8)
        self.assertTrue(report['evicted'])
    def test_rare_win_protection_is_revoked_by_regular_use(self):
        g=Game('rare',motifs=('r',),winning_motifs=frozenset({'r'}))
        _,r=select([g],[],policy=Policy(capacity=10))
        self.assertEqual(r['rare_protected'],['rare'])
        _,r=select([g],[],uses={'r':9},policy=Policy(capacity=10))
        self.assertEqual(r['rare_protected'],[])
    def test_common_win_is_not_rare(self):
        gs=[Game(str(i),motifs=('x',),winning_motifs=frozenset({'x'})) for i in range(4)]
        _,r=select(gs,[],policy=Policy(capacity=10))
        self.assertFalse(r['rare_protected'])
    def test_pin_survives_duplicate_and_too_many_humans_is_rejected(self):
        g=Game('h',True)
        kept,_=select([g],[Game('h')]);self.assertTrue(kept[0].human)
        with self.assertRaises(ValueError):select([g,Game('h2',True)],[],policy=Policy(capacity=1))
    def test_rare_reserve_is_bounded(self):
        gs=[Game(str(i),motifs=(str(i),),winning_motifs=frozenset({str(i)})) for i in range(100)]
        kept,r=select([],gs,policy=Policy(capacity=10))
        self.assertEqual(len(kept),10);self.assertEqual(len(r['rare_protected']),1)
if __name__=='__main__':unittest.main()
