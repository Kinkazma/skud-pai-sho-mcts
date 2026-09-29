#!/usr/bin/env python3
"""Isolated NumPy diagnostic; never publishes into a campaign.

Matches native tanh value/shared linear legal-action policy and SGD. Width growth
preserves the initial function. Read-only frozen native feature/search exports.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def read_examples(root):
    path=Path(root)/'examples.jsonl'
    if path.exists():
        with path.open() as stream:return [json.loads(line) for line in stream]
    with gzip.open(str(path)+'.gz','rt') as stream:return [json.loads(line) for line in stream]


class Network:
    def __init__(self, parameters, width=32, seed=0):
        p=np.asarray(parameters,dtype=np.float64)
        old=[p[:4096].reshape(32,128),p[4096:4128],p[4128:4160],p[4160:4161],p[4161:5185].reshape(32,32),p[5185:]]
        assert width in (32,64,128)
        copies=width//32;rng=np.random.default_rng(seed)
        self.w=[np.repeat(old[0],copies,axis=0),np.repeat(old[1],copies),
                np.repeat(old[2],copies)/copies,old[3].copy(),
                np.repeat(old[4],copies,axis=1)/copies,old[5].copy()]
        if copies>1:
            for ix in (2,4):
                shape=(32,copies) if ix==2 else (32,32,copies)
                noise=rng.normal(0,.001,shape);noise-=noise.mean(axis=-1,keepdims=True)
                self.w[ix]+=noise.reshape(self.w[ix].shape)
    def flat(self):return np.concatenate([w.ravel() for w in self.w])
    def forward(self,b):
        w,bias,v,vb,c,cb=self.w
        h=np.tanh(b['x']@w.T+bias);value=np.tanh(h@v+vb[0]);context=h@c.T+cb
        logits=np.einsum('ij,ij->i',b['a'],context[b['segment']])
        maximum=np.maximum.reduceat(logits,b['starts'])
        exp=np.exp(logits-maximum[b['segment']]);normal=np.add.reduceat(exp,b['starts'])
        prob=exp/normal[b['segment']]
        return h,value,logits,prob,maximum+np.log(normal)
    def gradient(self,b):
        h,value,logits,prob,logz=self.forward(b);n=len(value)
        w,bias,v,vb,c,cb=self.w
        dv=(value-b['value'])*(1-value**2)/n
        dl=(prob-b['policy'])*b['weight'][b['segment']]/n
        dc=np.add.reduceat(dl[:,None]*b['a'],b['starts'],axis=0)
        dh=(dv[:,None]*v+dc@c)*(1-h*h)
        g=[dh.T@b['x'],dh.sum(axis=0),h.T@dv,np.array([dv.sum()]),dc.T@h,dc.sum(axis=0)]
        ce=np.add.reduceat(b['policy']*(logz[b['segment']]-logits),b['starts'])
        return g,(.5*np.mean((value-b['value'])**2),float(ce.mean()))
    def step(self,b,rate):
        g,_=self.gradient(b);g=[a+1e-5*w for a,w in zip(g,self.w)]
        norm=np.sqrt(sum(np.sum(a*a) for a in g));scale=min(1,10/max(norm,1e-300))
        self.w=[w-rate*scale*a for w,a in zip(self.w,g)]

def pack(rows):
    lengths=np.array([len(r['actions']) for r in rows]);starts=np.r_[0,np.cumsum(lengths)[:-1]]
    return dict(x=np.array([r['state'] for r in rows]),a=np.concatenate([r['actions'] for r in rows]),
                policy=np.concatenate([r['policy'] for r in rows]),value=np.array([r['value'] for r in rows]),
                weight=np.array([r['policy_weight'] for r in rows]),starts=starts,
                segment=np.repeat(np.arange(len(rows)),lengths))

def evaluate(net,rows):
    b=pack(rows);h,value,logits,prob,logz=net.forward(b)
    ce=np.add.reduceat(b['policy']*(logz[b['segment']]-logits),b['starts'])
    entropy=np.add.reduceat(-b['policy']*np.log(np.maximum(b['policy'],1e-300)),b['starts'])
    mass=[];agreements=[];ranks=[]
    for i,start in enumerate(b['starts']):
        end=b['starts'][i+1] if i+1<len(rows) else len(logits);p=b['policy'][start:end];l=logits[start:end]
        chosen=int(np.argmax(l));mass.append(p[chosen]);agreements.append(p[chosen]>=p.max()-1e-12)
        best=np.flatnonzero(p>=p.max()-1e-12);ranks.append(1+min(np.count_nonzero(l>l[j]+1e-12) for j in best))
    return dict(value_mse=float(np.mean((value-b['value'])**2)),policy_kl=float(np.mean(ce-entropy)),
                policy_ce=float(ce.mean()),top_target_agreement=float(np.mean(agreements)),
                selected_target_mass=float(np.mean(mass)),mean_target_rank=float(np.mean(ranks)),
                hidden_saturated_fraction=float(np.mean(abs(h)>.99)),examples=len(rows))

def train_stage(net,rows,groups,rng,rate,seconds,rehearsal=None):
    start=time.perf_counter();steps=used=0;curves=[]
    while time.perf_counter()-start<seconds:
        batch=[rows[i] for i in rng.integers(len(rows),size=16)]
        if rehearsal:
            for i in range(len(batch)):
                if rng.random()<.1:batch[i]=rehearsal[rng.integers(len(rehearsal))]
        net.step(pack(batch),rate);steps+=1;used+=len(batch)
        if steps in (1,32,128,512,2048):
            curves.append(dict(steps=steps,seconds=time.perf_counter()-start,metrics={k:evaluate(net,v) for k,v in groups.items()}))
    elapsed=time.perf_counter()-start
    return dict(steps=steps,samples=used,seconds=elapsed,curves=curves,metrics={k:evaluate(net,v) for k,v in groups.items()})

def worker(args):
    width,root,seconds,rate=args;root=Path(root)
    rows=read_examples(root);groups={k:[r for r in rows if r['group']==k] for k in ('learn','interference','heldout')}
    parameters=json.loads((root/'model.json').read_text())['parameters'];results=[]
    for seed in (11,29,47):
        net=Network(parameters,width,seed);rng=np.random.default_rng(seed)
        initial={k:evaluate(net,v) for k,v in groups.items()}
        a=train_stage(net,groups['learn'],groups,rng,rate,seconds)
        np.savez(root/f'width-{width}-seed-{seed}-assimilated.npz',*net.w)
        no_replay=copy.deepcopy(net);with_replay=copy.deepcopy(net)
        b=train_stage(no_replay,groups['interference'],groups,np.random.default_rng(seed+100),rate,seconds)
        c=train_stage(with_replay,groups['interference'],groups,np.random.default_rng(seed+100),rate,seconds,groups['learn'])
        result=dict(width=width,seed=seed,parameters=len(net.flat()),initial=initial,assimilation=a,interference=b,rehearsal=c)
        results.append(result);(root/f'width-{width}.json').write_text(json.dumps(results,indent=2))
        print(f'width {width} seed {seed} finished: KL {initial["learn"]["policy_kl"]:.3f} -> {a["metrics"]["learn"]["policy_kl"]:.3f}',flush=True)
    return results

def diagnostics(rows,parameters):
    parity=[]
    for r in rows:
        ref=r['native_parity']
        if not ref:continue
        n=Network(parameters);b=pack([r]);_,value,logits,_,_=n.forward(b);g,loss=n.gradient(b)
        np.testing.assert_allclose(value[0],ref['value'],atol=1e-12)
        np.testing.assert_allclose(logits,ref['logits'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(np.concatenate([a.ravel()for a in g]),ref['gradient'],atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(loss,[ref['loss_value'],ref['loss_policy']],atol=1e-10)
        n.step(b,.001);np.testing.assert_allclose(n.flat(),ref['after_step'],atol=1e-12,rtol=1e-10)
        base=Network(parameters).forward(b)
        for width in (64,128):
            grow=Network(parameters,width,11).forward(b)
            np.testing.assert_allclose(grow[1],base[1],atol=1e-12)
            np.testing.assert_allclose(grow[2],base[2],atol=1e-10)
        parity.append(r['bundle'])
    searches={};tactics=[r for r in rows if r['immediate_wins']]
    for budget in ('policy',256,512):
        picks=[r['policy_pick'] if budget=='policy' else next(s['selected'] for s in r['searches']if s['budget']==budget)for r in rows]
        searches[str(budget)]=dict(top_target_agreement=float(np.mean([r['policy'][i]>=max(r['policy'])-1e-12 for r,i in zip(rows,picks)])),
            immediate_wins_found=sum(i in r['immediate_wins'] for r,i in zip(rows,picks) if r['immediate_wins']),tactical_positions=len(tactics))
        if budget!='policy':
            times=[next(s['seconds'] for s in r['searches'] if s['budget']==budget)for r in rows]
            searches[str(budget)].update(mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)))
    # Identical action features are an exact representation limit at every width.
    collisions=[]
    for r in rows:
        a=np.asarray(r['actions']);_,inverse=np.unique(a,axis=0,return_inverse=True)
        maxima=[];best=max(r['policy']);best_ids=np.flatnonzero(np.array(r['policy'])>=best-1e-12)
        for i in best_ids:
            tied=np.flatnonzero(inverse==inverse[i]);maxima.append(len(tied))
        collisions.append(dict(actions=len(a),unique=len(set(inverse)),best_target_aliases=min(maxima)))
    return dict(parity_examples=len(parity),search=searches,action_aliases=collisions)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--seconds',type=float,default=20);args=parser.parse_args();root=args.root
    rows=read_examples(root);parameters=json.loads((root/'model.json').read_text())['parameters']
    report=diagnostics(rows,parameters);groups={k:[r for r in rows if r['group']==k]for k in ('learn','interference','heldout')}
    # Equal-update rate triage; assimilation only. Hold-out untouched.
    pilot=[]
    for rate in (.001,.01,.05):
        net=Network(parameters);rng=np.random.default_rng(91)
        for _ in range(128):net.step(pack([groups['learn'][i] for i in rng.integers(len(groups['learn']),size=16)]),rate)
        metric=evaluate(net,groups['learn']);pilot.append(dict(rate=rate,metrics=metric))
    rate=min(pilot,key=lambda p:p['metrics']['policy_kl']+.5*p['metrics']['value_mse'])['rate'];report['pilot']=pilot;report['rate']=rate
    (root/'diagnostics.json').write_text(json.dumps(report,indent=2));print('native parity passed; rate',rate,flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(worker,[(w,str(root),args.seconds,rate)for w in (32,64,128)]))
    report['experiments']=results;(root/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
