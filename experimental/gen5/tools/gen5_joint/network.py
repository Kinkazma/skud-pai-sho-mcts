"""Shared action hidden state: policy and masked/count consequence supervision."""
import numpy as np
from gen5_r2 import graph
from gen5_r2.network import Network, prepare as base_prepare, softmax


def prepare(row):
    r=base_prepare(row)
    r['counts']=np.array([y['counts'] for y in row['structured']],dtype=float)
    labels=[y['events'] for y in row['structured']]
    r['known']=np.array([[x is not None for x in a] for a in labels])
    r['events']=np.array([[False if x is None else x for x in a] for a in labels],dtype=float)
    r['rare']=np.any(r['known'] & (r['events']>0),axis=0)
    # Keep the numeric arrays once; the durable JSON remains on disk for audit.
    for key in ['structured','action_names','threat_provenance']:
        r.pop(key,None)
    return r


def consequence_loss(y,r):
    """Independent bounded budgets; balanced known classes per event head."""
    d=np.zeros_like(y)
    err=y[:,1:11]-r['counts']; abs_err=np.abs(err)
    count=.2*np.where(abs_err<=1,.5*err**2,abs_err-.5).mean()
    d[:,1:11]=.2*np.clip(err,-1,1)/err.size
    events=y[:,11:]; dy=np.zeros_like(events); event_loss=0.
    active=[j for j in range(10) if r['known'][:,j].any()]
    for j in active:
        classes=[r['known'][:,j] & (r['events'][:,j]==v) for v in [0,1]]
        classes=[mask for mask in classes if mask.any()]
        for mask in classes:
            scale=.2/(len(active)*len(classes)*int(mask.sum()))
            logits=events[mask,j]; target=r['events'][mask,j]
            event_loss+=scale*float((np.logaddexp(0,logits)-target*logits).sum())
            dy[mask,j]=scale*(np.exp(-np.logaddexp(0,-logits))-target)
    d[:,11:]=dy
    return float(count+event_loss),d


class JointNetwork(Network):
    def __init__(self,seed,mode,joint):
        super().__init__(seed,mode,'policy');self.joint=joint
        self.w['out']=np.zeros((64,21));self.w['out_b']=np.zeros(21)
        self.m={k:np.zeros_like(v) for k,v in self.w.items()}
        self.v={k:np.zeros_like(v) for k,v in self.w.items()}

    def loss_gradient(self,r,gradient=True):
        prediction,c=self.forward(r)
        pooled,nodes,gc,attention,query,context,h,y=c
        logits=r['base_logits']+y[:,0];dy=np.zeros_like(y)
        if r['winning'].any():
            loss=np.logaddexp.reduce(logits)-np.logaddexp.reduce(logits[r['winning']])
            dy[:,0]=softmax(logits);dy[r['winning'],0]-=softmax(logits[r['winning']])
        else:
            # No proof of a best move: preserve the historical policy, not the human action.
            logp=logits-np.logaddexp.reduce(logits)
            loss=.1*float((r['prior']*(r['base_logits']-logp)).sum())
            dy[:,0]=.1*(softmax(logits)-r['prior'])
        if self.joint:
            aux,da=consequence_loss(y,r);loss+=aux;dy+=da
        if not gradient:return float(loss)
        dw={k:np.zeros_like(v) for k,v in self.w.items()}
        dw['out']=h.T@dy;dw['out_b']=dy.sum(axis=0)
        dh=(dy@self.w['out'].T)*(1-h*h);total=dh.sum(axis=0)
        dw['hidden'][:417]=np.outer(r['state'],total)
        dw['hidden'][417:449]=r['actions'].T@dh
        dw['hidden'][449:481]=np.outer(pooled,total)
        dw['hidden'][481:]=context.T@dh;dw['hidden_b']=total
        dpool=self.w['hidden'][449:481]@total;dnodes=np.zeros_like(nodes)
        if attention is not None:
            dc=dh@self.w['hidden'][481:].T;dnodes+=attention.T@dc
            da=dc@nodes.T;ds=attention*(da-(da*attention).sum(axis=1,keepdims=True))
            dq=ds@nodes/4;dnodes+=ds.T@query/4;dw['query']=r['actions'].T@dq
        dw['encoder']=graph.backward(r,self.w['encoder'],self.mode,gc,dpool,dnodes)
        return float(loss),dw
