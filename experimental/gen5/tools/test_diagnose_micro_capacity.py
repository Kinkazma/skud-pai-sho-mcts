import unittest
import numpy as np
from diagnose_micro_capacity import Network,pack

class CapacityTests(unittest.TestCase):
    def test_widening_preserves_function_and_has_distinct_gradients(self):
        rng=np.random.default_rng(2);parameters=rng.normal(0,.1,5217)
        row=dict(state=rng.normal(size=128),actions=rng.normal(size=(7,32)),policy=np.ones(7)/7,value=.3,policy_weight=1.)
        batch=pack([row]);base=Network(parameters).forward(batch)
        for width in (64,128):
            model=Network(parameters,width);current=model.forward(batch)
            np.testing.assert_allclose(current[1],base[1],atol=1e-12)
            np.testing.assert_allclose(current[2],base[2],atol=1e-12)
            g,_=model.gradient(batch)
            self.assertFalse(np.array_equal(g[0][0],g[0][1]))

    def test_wide_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(4);net=Network(rng.normal(0,.1,5217),64)
        rows=[dict(state=rng.normal(size=128),actions=rng.normal(size=(n,32)),policy=np.ones(n)/n,value=.1,policy_weight=.7)for n in (3,7)]
        batch=pack(rows);grad,_=net.gradient(batch)
        def objective():
            _,v,l,_,z=net.forward(batch)
            ce=np.add.reduceat(batch['policy']*(z[batch['segment']]-l),batch['starts'])
            return .5*np.mean((v-batch['value'])**2)+np.mean(batch['weight']*ce)
        for w,g in zip(net.w,grad):
            for i in set([0,w.size//2,w.size-1]):
                old=w.flat[i];w.flat[i]=old+1e-6;plus=objective();w.flat[i]=old-1e-6;minus=objective();w.flat[i]=old
                self.assertAlmostEqual(g.flat[i],(plus-minus)/2e-6,places=7)

    def test_batch_loss_gradient_equals_mean_of_examples(self):
        rng=np.random.default_rng(7);net=Network(rng.normal(0,.1,5217))
        rows=[dict(state=rng.normal(size=128),actions=rng.normal(size=(n,32)),policy=np.ones(n)/n,value=.2,policy_weight=1.)for n in (2,5)]
        batch,_=net.gradient(pack(rows));single=[net.gradient(pack([r]))[0] for r in rows]
        for i,g in enumerate(batch):np.testing.assert_allclose(g,(single[0][i]+single[1][i])/2,atol=1e-12)

if __name__=='__main__':unittest.main()
