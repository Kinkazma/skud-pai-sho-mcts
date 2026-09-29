#!/usr/bin/env python3
"""Additional frozen action-head/inference diagnostics and static result chart."""
from diagnose_micro_capacity import Network,pack,evaluate,read_examples
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

def analyze(root):
    rows=read_examples(root);parameters=json.loads((root/'model.json').read_text())['parameters'];base=Network(parameters)
    oracle=[]
    for row in rows:
        a=np.asarray(row['actions']);target=np.asarray(row['policy']);entropy=-np.sum(target*np.log(np.maximum(target,1e-300)))
        h=base.forward(pack([row]))[0][0];context=base.w[4]@h+base.w[5]
        def objective(c):
            logits=a@c;v=logits.max();exp=np.exp(logits-v);z=exp.sum();p=exp/z
            return float(v+np.log(z)-target@logits),a.T@(p-target)
        result=minimize(objective,context,jac=True,method='L-BFGS-B',options={'maxiter':120,'gtol':1e-7,'ftol':1e-10})
        before=objective(context)[0]-entropy;after=objective(result.x)[0]-entropy
        oracle.append(dict(group=row['group'],source=row['source'],initial_kl=before,free_context_kl=after,converged=bool(result.success),iterations=int(result.nit)))
    # Batched CPU inference comparison; warm data. This is NOT native MCTS timing.
    b=pack(rows);timings={}
    for width in (32,64,128):
        net=Network(parameters,width,11);net.forward(b);times=[]
        for _ in range(30):
            t=time.perf_counter();net.forward(b);times.append(time.perf_counter()-t)
        timings[str(width)]={'median_seconds_per_position':float(np.median(times)/len(rows)),'parameters':len(net.flat())}
    report={'linear_head_oracle':oracle,'numpy_batched_inference':timings}
    (root/'head-diagnostic.json').write_text(json.dumps(report,indent=2))
    print('free-context KL',np.mean([r['initial_kl']for r in oracle]),'->',np.mean([r['free_context_kl']for r in oracle]),flush=True)


def tactical_feasibility(root):
    from scipy.optimize import linprog
    from fractions import Fraction
    rows=read_examples(root);results=[];certificates=[]
    for row in rows:
        wins=row['immediate_wins']
        if not wins:continue
        a=np.array(row['actions']);non=[i for i in range(len(a))if i not in wins];best=-1.;statuses=[]
        for i in wins:
            result=linprog(np.r_[np.zeros(32),-1.],A_ub=np.c_[a[non]-a[i],np.ones(len(non))],b_ub=np.zeros(len(non)),bounds=[(-1,1)]*32+[(None,None)],method='highs')
            statuses.append(int(result.status))
            if result.success:best=max(best,float(result.x[-1]))
            if best>1e-8:break
        results.append(dict(source=row['source'],decision=row['decision'],best_tested_margin=best,strict_winning_preference_possible=best>1e-8,statuses=statuses,search512_wins=row['searches'][1]['selected']in wins))
        if best<=1e-8 and all(status==0 for status in statuses):
            proofs=[]
            for i in wins:
                result=linprog(np.zeros(len(non)),A_eq=np.r_[a[non].T,np.ones((1,len(non)))],b_eq=np.r_[a[i],1.],bounds=(0,None),method='highs')
                if not result.success:continue
                support=[(non[j],Fraction(float(w)).limit_denominator(1000000))for j,w in enumerate(result.x)if w>1e-9]
                exact=sum(w for j,w in support)==1 and all(sum(w*Fraction(float(a[j,k]))for j,w in support)==Fraction(float(a[i,k]))for k in range(32))
                proofs.append(dict(winning_action=row['action_names'][i],winning_features=a[i].tolist(),support=[dict(action=row['action_names'][j],weight=str(w),features=a[j].tolist())for j,w in support],exact_rational_identity=exact))
            certificates.append(dict(source=row['source'],decision=row['decision'],certificates=proofs))
    (root/'linear-tactical-feasibility.json').write_text(json.dumps(results,indent=2))
    (root/'linear-impossibility-certificates.json').write_text(json.dumps(certificates,indent=2))


def chart(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((root/'results.json').read_text());head=json.loads((root/'head-diagnostic.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors=['#387db5','#c98e23','#319270'];xs=np.arange(3)
    for ix,runs in enumerate(results['experiments']):
        width=runs[0]['width'];means=[];std=[]
        for stage in ('initial','assimilation','interference','rehearsal'):
            values=[(r[stage]if stage=='initial'else r[stage]['metrics'])['learn']['policy_kl']for r in runs]
            means.append(np.mean(values));std.append(np.std(values))
        axes[0,0].errorbar(range(4),means,yerr=std,marker='o',linestyle='none',label=f'{width} hidden',color=colors[ix])
        held=[r['assimilation']['metrics']['heldout']['policy_kl']for r in runs]
        axes[0,1].bar(ix,np.mean(held),yerr=np.std(held),color=colors[ix])
    axes[0,0].set_xticks(range(4),['Initial','Learn A','A → B\nno rehearsal','A → B\n10% A rehearsal']);axes[0,0].set_ylabel('KL to stored targets (lower is better)');axes[0,0].set_title('Assimilation and retention on A');axes[0,0].legend()
    axes[0,1].axhline(results['experiments'][0][0]['initial']['heldout']['policy_kl'],color='black',linestyle='--',label='Before fitting');axes[0,1].legend();axes[0,1].set_xticks(xs,['32','64','128']);axes[0,1].set_title('Other sources excluded from diagnostic fitting');axes[0,1].set_ylabel('Policy KL after learning A')
    search=results['search'];names=['policy','256','512'];hits=[search[n]['immediate_wins_found']for n in names];n=search['policy']['tactical_positions']
    axes[1,0].bar(xs,hits,color=colors);axes[1,0].set_xticks(xs,['Policy only','MCTS 256','MCTS 512']);axes[1,0].set_ylim(0,n+1);axes[1,0].set_title(f'Immediate regulatory wins found / {n}')
    for i,v in enumerate(hits):axes[1,0].text(i,v+.15,str(v),ha='center')
    oracle=head['linear_head_oracle'];a=np.mean([r['initial_kl']for r in oracle]);b=np.mean([r['free_context_kl']for r in oracle]);axes[1,1].bar([0,1],[a,b],color=colors[:2]);axes[1,1].set_xticks([0,1],['Current network','Free context per position']);axes[1,1].set_title('Remaining error in the linear action head');axes[1,1].set_ylabel('Mean policy KL')
    fig.suptitle('Gen5 frozen diagnostic · 96 human sources / 171 positions\nTarget fidelity and immediate tactics, not Elo or full-game strength',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(root/'diagnostic.png',dpi=170);fig.savefig(root/'diagnostic.svg');plt.close(fig)
    svg=root/'diagnostic.svg';svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--chart',action='store_true');p.add_argument('--tactics',action='store_true');a=p.parse_args()
    if a.chart:chart(a.root)
    elif a.tactics:tactical_feasibility(a.root)
    else:analyze(a.root)
