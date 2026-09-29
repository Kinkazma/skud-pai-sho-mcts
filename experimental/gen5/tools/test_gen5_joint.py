"""Derivative and missing-evidence contracts for joint supervision."""
import unittest
import numpy as np
from gen5_joint.network import JointNetwork, consequence_loss
from gen5_joint.r1_control import ranking, defence_choices
from test_gen5_r2 import sample


class JointTests(unittest.TestCase):
    def test_defence_unknown_is_never_counted_as_safe(self):
        r={'winning':np.zeros(3,dtype=bool),'known':np.zeros((3,10),dtype=bool),
           'events':np.zeros((3,10))}
        r['known'][:2,5]=True;r['events'][1,5]=1
        result=defence_choices([r,r,r],list(np.eye(3)))
        self.assertEqual(result,{'safe':1,'unsafe':1,'unknown':1,'eligible_roots':3})

    def test_ranking_keeps_threshold_ties_and_unknown_heads(self):
        tied = ranking([np.array([0., 0., 0., 0.])], [np.array([1, 0, 1, 0])])
        self.assertEqual(tied['roc_auc'], .5)
        self.assertEqual(tied['average_precision'], .5)
        perfect = ranking([np.array([1., 0., 1., 0.])], [np.array([1, 0, 1, 0])])
        self.assertEqual(perfect['roc_auc'], 1.)
        self.assertEqual(perfect['average_precision'], 1.)
        absent = ranking([np.array([])], [np.array([])])
        self.assertIsNone(absent['roc_auc'])
        self.assertIsNone(absent['average_precision'])
        self.assertEqual(absent['positive'] + absent['negative'], 0)

    def test_mask_and_rare_balance(self):
        r={'counts':np.zeros((10,10)),'known':np.zeros((10,10),dtype=bool),'events':np.zeros((10,10))}
        y=np.zeros((10,21));r['known'][:,0]=True;r['events'][0,0]=1
        loss,g=consequence_loss(y,r)
        self.assertAlmostEqual(loss,.2*np.log(2))
        self.assertAlmostEqual(g[0,11],-.05)
        self.assertAlmostEqual(g[1:,11].sum(),.05)
        self.assertTrue((g[:,12:]==0).all())
        r['known'][:]=False
        self.assertEqual(consequence_loss(y,r)[0],0.)

    def test_joint_gradient(self):
        r=sample('policy');n=len(r['actions'])
        r['counts']=np.tile(np.linspace(-.4,.4,10),(n,1))
        r['events']=np.tile(np.arange(10)%2,(n,1));r['known']=np.ones((n,10),dtype=bool)
        r['known'][:,5]=False
        for mode in ['dense','messages']:
            net=JointNetwork(17,mode,True)
            net.w['out'][:]=np.random.default_rng(1).normal(0,.03,net.w['out'].shape)
            _,g=net.loss_gradient(r)
            for k in net.w:
                for i in sorted({0,net.w[k].size//2,net.w[k].size-1}):
                    a=net.w[k].flat[i];eps=1e-5
                    net.w[k].flat[i]=a+eps;hi=net.loss_gradient(r,False)
                    net.w[k].flat[i]=a-eps;lo=net.loss_gradient(r,False)
                    net.w[k].flat[i]=a
                    self.assertAlmostEqual((hi-lo)/(2*eps),g[k].flat[i],delta=2e-7,msg=f'{mode}/{k}/{i}')


if __name__=='__main__':unittest.main()
