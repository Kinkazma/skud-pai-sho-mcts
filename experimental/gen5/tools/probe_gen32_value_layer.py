#!/usr/bin/env python3
"""Small antisymmetric value fits, source-disjoint; never promotes a model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import json, sys, time
import numpy as np
raw=json.load(open(sys.argv[1])); rows=raw['rows']; records=[]
x=np.asarray([r['x'] for r in rows]);y=np.asarray([r['y'] for r in rows]); test=np.asarray([r['held_out'] for r in rows]); groups=np.asarray([r['source'] for r in rows])
# Same position, opposite value perspective: differences negate; absolute seats swap.
opp=x.copy();opp[:,:64]*=-1;opp[:,64:76]=x[:,76:88];opp[:,76:88]=x[:,64:76];opp[:,88:106]=x[:,106:124];opp[:,106:124]=x[:,88:106]
train=np.flatnonzero(~test)
def mse(pred,mask):
 return float(np.mean([np.mean((pred[(groups==g)&mask]-y[(groups==g)&mask])**2) for g in np.unique(groups[mask])]))
for inputs,hidden in [(64,0),(64,16),(128,16)]:
 for seed in [17,731,1911]:
  rng=np.random.default_rng(seed); xx=x[:,:inputs];oo=opp[:,:inputs]
  if hidden: params=[rng.normal(0,.15,(inputs,hidden)),np.zeros(hidden),np.zeros(hidden),np.zeros(1)]
  else: params=[np.zeros(inputs)]
  moments=[np.zeros_like(p) for p in params];squares=[np.zeros_like(p) for p in params]
  def forward(a,b):
   if not hidden:
    raw=(a-b)@params[0]/2;return np.tanh(raw),None
   h=np.tanh(a@params[0]+params[1]);k=np.tanh(b@params[0]+params[1]);raw=(h-k)@params[2]/2
   return np.tanh(raw),(h,k)
  started=time.perf_counter()
  for step in range(1,2001):
   idx=rng.choice(train,size=min(128,len(train)),replace=False);a,b=xx[idx],oo[idx]; pred,cache=forward(a,b)
   delta=(pred-y[idx])*(1-pred*pred)/len(idx)
   if not hidden: grads=[(a-b).T@delta/2]
   else:
    h,k=cache; dh=delta[:,None]*params[2]*(1-h*h)/2;dk=-delta[:,None]*params[2]*(1-k*k)/2
    grads=[a.T@dh+b.T@dk,(dh+dk).sum(0),(h-k).T@delta/2,np.zeros(1)]
   for i,g in enumerate(grads):
    g+=1e-5*params[i];moments[i]=.9*moments[i]+.1*g;squares[i]=.999*squares[i]+.001*g*g
    params[i]-=.003*(moments[i]/(1-.9**step))/(np.sqrt(squares[i]/(1-.999**step))+1e-8)
  seconds=time.perf_counter()-started;pred,_=forward(xx,oo)
  record={'inputs':inputs,'hidden':hidden,'seed':seed,'steps':2000,'fit_seconds':seconds,'train_game_mse':mse(pred,~test),'heldout_game_mse':mse(pred,test),'antisymmetry_max_error':float(np.max(np.abs(pred+forward(oo,xx)[0])))}
  records.append(record);print(json.dumps(record),flush=True)
json.dump({'source':sys.argv[1],'games':raw['games'],'positions':len(rows),'note':'Architecture triage, no MCTS strength/inference speed claim; frozen corpus split retained. All fits use 2000 updates.','results':records},open(sys.argv[2],'w'),indent=2)
