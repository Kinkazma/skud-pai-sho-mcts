"""Predeclared four-arm joint comparison with source groups and 50% recall."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .network import JointNetwork,prepare


class Sampler:
    def __init__(self,rows):
        def grouped(selected):
            d=collections.defaultdict(list)
            for r in selected:d[r['group']].append(r)
            return [d[k] for k in sorted(d)]
        self.all=grouped(rows)
        self.rare=[grouped([r for r in rows if r['rare'][j]]) for j in range(10)]
        self.rare=[a for a in self.rare if a]

    def pick(self,rng):
        groups=self.rare[int(rng.integers(len(self.rare)))] if self.rare and rng.random()<.5 else self.all
        rows=groups[int(rng.integers(len(groups)))];return rows[int(rng.integers(len(rows)))]


def measure(net,rows,auxiliary=False):
    result=[]
    for r in rows:
        prediction,cache=net.forward(r)
        item={'key':r['key'],'group':r['group'],'has_win':bool(r['winning'].any()),
              'win':bool(r['winning'][prediction.argmax()]),'winning_mass':float(prediction[r['winning']].sum())}
        if auxiliary:
            y=cache[-1];pred=y[:,11:]>=0;truth=r['events']>0;known=r['known']
            item['events']={'tp':(known&truth&pred).sum(axis=0).tolist(),
                            'fp':(known&~truth&pred).sum(axis=0).tolist(),
                            'tn':(known&~truth&~pred).sum(axis=0).tolist(),
                            'fn':(known&truth&~pred).sum(axis=0).tolist()}
            item['count_mae']=np.abs(y[:,1:11]-r['counts']).mean(axis=0).tolist()
        result.append(item)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root
    plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'corpus/summary.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(root/'plan.json')==summary['plan_sha256']
    assert sha(root/'corpus/rows.jsonl')==summary['rows_sha256']
    loaded=time.perf_counter()
    rows=[prepare(json.loads(s)) for s in (root/'corpus/rows.jsonl').open()]
    loading=time.perf_counter()-loaded
    splits={s:[r for r in rows if r['split']==s] for s in ['train','dev','final']}
    groups={s:{r['group'] for r in v} for s,v in splits.items()}
    assert not groups['train']&groups['dev'] and not groups['train']&groups['final'] and not groups['dev']&groups['final']
    ag=set(sorted(groups['train'])[::2]);a=[r for r in splits['train'] if r['group'] in ag];b=[r for r in splits['train'] if r['group'] not in ag]
    samplers=[Sampler(a),Sampler(b)];out=root/'learning';out.mkdir(exist_ok=False)
    counts={s:{'roots':len(v),'groups':len(groups[s]),'winning_roots':sum(r['winning'].any() for r in v)} for s,v in splits.items()}
    counts={s:{k:int(n) for k,n in v.items()} for s,v in counts.items()}
    (out/'split.json').write_text(json.dumps({'a_groups':sorted(ag),'counts':counts,'loading_seconds':loading},indent=2)+'\n')
    start=time.perf_counter();result={'plan_sha256':summary['plan_sha256'],'counts':counts,'trials':[]}
    for seed in plan['seeds']:
        for arm in plan['arms']:
            mode,kind=arm.split('-');net=JointNetwork(seed,mode,kind=='joint');rng=np.random.default_rng(seed+6000)
            trial={'seed':seed,'arm':arm,'curve':[]};t=time.perf_counter()
            for phase in ['A','B']:
                for step in range(plan['steps_per_phase']+1):
                    if step in plan['read_steps']:
                        point={'phase':phase,'steps':net.steps,
                               'old':measure(net,a),'new':measure(net,b),'dev':measure(net,splits['dev'])}
                        trial['curve'].append(point);net.save(out/f'{seed}-{arm}-{phase}-{step}.npz')
                        (out/f'{seed}-{arm}.json').write_text(json.dumps(trial,indent=2)+'\n')
                        print(json.dumps({'seed':seed,'arm':arm,'phase':phase,'step':step,
                                          'dev_win':sum(r['win'] for r in point['dev'])}),flush=True)
                    if step==plan['steps_per_phase']:break
                    net.update([samplers[0].pick(rng),samplers[0 if phase=='A' else 1].pick(rng)],rate=plan['rate'])
            trial['final']=measure(net,splits['final'],auxiliary=True)
            trial['seconds']=time.perf_counter()-t;result['trials'].append(trial)
            (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['seconds']=time.perf_counter()-start
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'complete':True,'trials':len(result['trials']),'seconds':result['seconds']},indent=2)+'\n')


if __name__=='__main__':main()
