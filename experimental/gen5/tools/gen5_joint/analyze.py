"""Paired source-group analysis; repeated initializations are not independent games."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np


def main(root):
    result=json.loads((root/'learning/results.json').read_text())
    baseline={}
    for line in (root/'corpus/rows.jsonl').open():
        r=json.loads(line)
        if r['split']=='final':
            baseline[r['key']]={'group':r['group'],'has_win':any(r['winning']),
                'win':r['winning'][int(np.argmax(r['prior']))]}
    trials=[]
    for t in result['trials']:
        a=[p for p in t['curve'] if p['phase']=='A'][-1]['old']
        before=t['curve'][0]['old'];end=t['curve'][-1]['old']
        gained={r['key'] for r,b in zip(a,before) if r['win'] and not b['win']}
        sums={k:np.sum([r['events'][k] for r in t['final']],axis=0) for k in ['tp','fp','tn','fn']}
        with np.errstate(invalid='ignore',divide='ignore'):
            precision=np.divide(sums['tp'],sums['tp']+sums['fp'],out=np.zeros(10),where=(sums['tp']+sums['fp'])>0)
            recall=np.divide(sums['tp'],sums['tp']+sums['fn'],out=np.zeros(10),where=(sums['tp']+sums['fn'])>0)
        trials.append({'seed':t['seed'],'arm':t['arm'],'seconds':t['seconds'],
            'final_wins':sum(r['win'] for r in t['final']),
            'old_a_gains':len(gained),'old_a_gains_retained':sum(r['win'] for r in end if r['key'] in gained),
            'final_events':{k:v.tolist() for k,v in sums.items()},'precision':precision.tolist(),'recall':recall.tolist()})
    arms={arm:[t for t in result['trials'] if t['arm']==arm] for arm in ['dense-policy','dense-joint','messages-policy','messages-joint']}
    comparisons=[];rng=np.random.default_rng(20260926)
    for a,b in [('dense-policy','messages-policy'),('dense-joint','messages-joint'),('messages-policy','messages-joint'),('dense-policy','dense-joint')]:
        by_group=defaultdict(list)
        for key,row in baseline.items():
            if not row['has_win']:continue
            va=np.mean([next(r['win'] for r in t['final'] if r['key']==key) for t in arms[a]])
            vb=np.mean([next(r['win'] for r in t['final'] if r['key']==key) for t in arms[b]])
            by_group[row['group']].append(vb-va)
        diffs=np.array([np.mean(v) for _,v in sorted(by_group.items())])
        boot=diffs[rng.integers(len(diffs),size=(8000,len(diffs)))].mean(axis=1)
        comparisons.append({'base':a,'candidate':b,'winning_source_groups':len(diffs),
            'group_weighted_delta':float(diffs.mean()),'exploratory_bonferroni95_interval':np.quantile(boot,[.00625,.99375]).tolist()})
    out={'scope':'Frozen-base sidecar joint policy/auxiliary, no native actor/value/publication in this comparison.',
        'baseline_final_wins':sum(r['win'] for r in baseline.values()),
        'final_winning_roots':sum(r['has_win'] for r in baseline.values()),'trials':trials,'comparisons':comparisons,
        'caution':'Group bootstrap is descriptive/exploratory; final panel now consulted, not untouched for later tuning.'}
    (root/'joint-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='trials'},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
