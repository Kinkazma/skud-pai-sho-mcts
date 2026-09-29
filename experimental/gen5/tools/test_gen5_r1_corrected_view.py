import copy
import unittest
from gen5_r1_corrected_view import corrected


class CorrectedViewTests(unittest.TestCase):
    def test_branch_preserves_outcome_proof_and_provenance(self):
        row = {'root': 'r', 'action': 'a', 'components_before_host_guest': [10, 7],
               'components_after_host_guest': [9, 7], 'outcome': 'win',
               'threat': {'status': 'terminal'}, 'held_out': True}
        old = copy.deepcopy(row)
        patch = {'root': 'r', 'action': 'a', 'old_before': [10, 7], 'old_after': [9, 7],
                 'corrected_before': [11, 7], 'corrected_after': [10, 7]}
        new = corrected(row, patch)
        self.assertEqual(row, old)
        self.assertEqual(new.pop('components_before_host_guest'), [11, 7])
        self.assertEqual(new.pop('components_after_host_guest'), [10, 7])
        del old['components_before_host_guest'], old['components_after_host_guest']
        self.assertEqual(new, old)
        for key in ['root', 'action', 'old_before', 'old_after']:
            wrong = dict(patch, **{key: 'wrong'})
            with self.assertRaises(ValueError):
                corrected(row, wrong)

    def test_root_preserves_piece_owner_and_rejects_stale_counts(self):
        row = {'key': 'r', 'position': {'components_host_guest': [10, 7],
                                      'pieces': [{'kind': 'L', 'owner': 'G'}]}}
        patch = {'root': 'r', 'old': [10, 7], 'corrected': [11, 7]}
        new = corrected(row, patch, root=True)
        self.assertEqual(new['position']['pieces'], row['position']['pieces'])
        self.assertEqual(new['position']['components_host_guest'], [11, 7])
        with self.assertRaises(ValueError):
            corrected(new, patch, root=True)


if __name__ == '__main__':
    unittest.main()
