import tempfile
import unittest
from pathlib import Path

import numpy as np
from gen5_relations_trial import Network, prepare


def example(proven=None):
    rng=np.random.default_rng(11)
    return prepare({"key":"synthetic","source":"synthetic","route":"ring","held_out":False,
        "state":rng.normal(0,.2,417).tolist(),"actions":rng.normal(0,.2,(6,32)).tolist(),
        "global":rng.normal(0,.2,20).tolist(),"edge_tokens":rng.normal(0,.2,(6,4,43)).tolist(),
        "prior":[.1,.2,.1,.1,.3,.2],"next_values":[.1,-.2,.2,.3,1.,-.1],
        "q_targets":[.3,None,-.1,.6,1.,.2],"terminal":[None,None,None,None,1.,None],
        "policy_target":[.1,.2,.1,.1,.3,.2] if proven is None else [0.,.5,0.,0.,.5,0.],
        "proven":proven,"auxiliary":rng.normal(0,.2,(6,12)).tolist()})


class Gradients(unittest.TestCase):
    def test_neutral_policy_and_q_are_exact_and_terminals_stay_exact(self):
        n=Network(7,True,.1);r=example();p,q,_,_=n.forward(r)
        np.testing.assert_array_equal(p,r['prior']);np.testing.assert_array_equal(q,r['next_values'])
        n.w['out_b'][1]=.2
        self.assertEqual(n.forward(r)[1][4],1.)

    def test_all_shared_blocks_and_both_policy_objectives(self):
        worst=0.
        for proven in [None,1]:
            n=Network(7,True,.1);n.w['out'][:,:2]=.02
            r=example(proven);scale=np.linspace(.2,.5,12)
            _,g=n.loss_gradient(r,scale)
            for key in n.w:
                i=np.unravel_index(np.argmax(np.abs(g[key])),g[key].shape)
                self.assertGreater(abs(g[key][i]),1e-9)
                original=n.w[key][i];eps=1e-5
                n.w[key][i]=original+eps;plus=n.loss_gradient(r,scale,False)
                n.w[key][i]=original-eps;minus=n.loss_gradient(r,scale,False)
                n.w[key][i]=original
                error=abs((plus-minus)/(2*eps)-g[key][i]);worst=max(error,worst)
                self.assertLess(error,2e-7,(key,error))
        print('maximum checked gradient error',worst)

    def test_edge_order_and_action_order_do_not_encode_labels(self):
        r=example(1);n=Network(7,True,.1);n.w['out'][:,:2]=.02
        original=n.forward(r)
        reordered=dict(r);reordered['tokens']=r['tokens'][:,::-1,:].copy()
        for a,b in zip(original[:3],n.forward(reordered)[:3]):np.testing.assert_allclose(a,b,atol=2e-15,rtol=2e-15)
        order=np.array([3,1,0,5,4,2]);reordered=dict(r)
        for k in ['tokens','prior','actions','next_values','q_targets','q_mask','terminal_mask','q_base_logit','policy_target','auxiliary','winning']:
            reordered[k]=r[k][order]
        for a,b in zip(original[:3],n.forward(reordered)[:3]):np.testing.assert_allclose(a[order],b,atol=2e-15,rtol=2e-15)

    def test_checkpoint_reproduces_next_optimizer_step(self):
        a=Network(7,True,.1);r=example();scale=np.ones(12);a.update([r],scale)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.npz';a.save(p)
            b=Network(7,True,.1)
            with np.load(p) as saved:
                for k in b.w:
                    b.w[k]=saved[k].copy();b.m[k]=saved['adam_m_'+k].copy();b.v[k]=saved['adam_v_'+k].copy()
                b.steps=int(saved['steps'])
            a.update([r],scale);b.update([r],scale)
            for k in a.w:np.testing.assert_array_equal(a.w[k],b.w[k])


if __name__=='__main__':unittest.main()
