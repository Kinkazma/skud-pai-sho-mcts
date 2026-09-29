import unittest
import numpy as np
from gen5_r2.network import Network, prepare


def sample(task='policy'):
    nodes = np.zeros((3, 16)); nodes[np.arange(3), [0, 6, 2]] = 1
    nodes[:, 12] = [1, 0, 1]; nodes[:, 13] = [0, 1, 0]
    nodes[:, 14:] = [[0, .5], [.5, .5], [-.5, -.5]]
    row = {'task': task, 'state': np.linspace(-.5, .5, 417).tolist(), 'nodes': nodes.tolist(),
           'edges': [{'source': 0, 'destination': 1, 'features': [1, 0, .25, 0, 1, 0]},
                     {'source': 1, 'destination': 0, 'features': [1, 0, -.25, 0, 1, 0]}],
           'actions': np.random.default_rng(17).normal(size=(4, 32)).tolist(),
           'prior': [.1, .2, .3, .4], 'winning': [False, True, False, True],
           'targets': [1, None, 0, None]}
    return prepare(row)


class R2Tests(unittest.TestCase):
    def test_neutral_policy_is_exact_and_isolated_piece_remains_visible(self):
        r = sample()
        for mode in ['dense', 'edges', 'pieces', 'messages']:
            net = Network(17, mode, 'policy')
            np.testing.assert_array_equal(net.forward(r)[0], r['prior'])

    def test_complete_winning_set_is_not_a_single_action_label(self):
        r = sample(); net = Network(17, 'messages', 'policy')
        loss, g = net.loss_gradient(r)
        self.assertAlmostEqual(loss, -np.log(.6))
        # A decision whose every legal action wins must receive no policy push.
        r['winning'][:] = True
        loss, g = net.loss_gradient(r)
        self.assertAlmostEqual(loss, 0.)
        self.assertTrue(all(np.max(np.abs(x)) < 1e-14 for x in g.values()))

    def test_attention_and_shared_encoder_gradients(self):
        for task in ['policy', 'motif']:
            r = sample(task)
            for mode in ['dense', 'edges', 'pieces', 'messages']:
                net = Network(29, mode, task)
                net.w['out'][:] = np.random.default_rng(7).normal(0, .2, net.w['out'].shape)
                _, grad = net.loss_gradient(r)
                rng = np.random.default_rng(43)
                for key, w in net.w.items():
                    indices = np.unique(np.r_[0, w.size-1, rng.integers(w.size, size=12)])
                    for k in indices:
                        x = w.flat[k]; w.flat[k] = x + 1e-6
                        plus = net.loss_gradient(r, False)
                        w.flat[k] = x - 1e-6
                        minus = net.loss_gradient(r, False); w.flat[k] = x
                        self.assertAlmostEqual((plus-minus)/2e-6, grad[key].flat[k], delta=2e-7,
                                               msg=f'{task}/{mode}/{key}/{k}')


if __name__ == '__main__':
    unittest.main()
