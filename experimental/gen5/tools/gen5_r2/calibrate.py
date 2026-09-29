"""Longer R2 learning curves using original training groups only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .network import Network, prepare
from .trial import group_sample, measure, pick, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path); parser.add_argument('out', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.native/'summary.json').read_text())
    if sha(args.native/'rows.jsonl') != summary['rows_sha256']:
        raise ValueError('Native frozen data changed')
    raw = [json.loads(s) for s in (args.native/'rows.jsonl').read_text().splitlines()]
    # No original held-out row is even prepared or measured in this calibration.
    train = [prepare(r) for r in raw if r['task']=='policy' and not r['held_out']]
    groups = sorted({r['group'] for r in train}, key=lambda g: hashlib.sha256(('R2-inner-v1:'+g).encode()).hexdigest())
    validation_groups = set(groups[:max(1,len(groups)//5)])
    fit_groups = [g for g in groups if g not in validation_groups]
    a_groups = set(fit_groups[::2])
    dev = [r for r in train if r['group'] in validation_groups]
    a = [r for r in train if r['group'] in a_groups]
    b = [r for r in train if r['group'] not in validation_groups|a_groups]
    ag, bg = group_sample(a), group_sample(b)
    assert a and b and dev
    args.out.mkdir(exist_ok=False)
    plan = {'schema':'gen5-r2-calibration-v1','seeds':[17,29,43],'arms':['dense','messages'],
            'checkpoints':[0,2048,8192,16384], 'steps_per_phase':16384,
            'phase_B_recall':.5,'native_rows_sha256':summary['rows_sha256'],
            'a_groups':sorted(a_groups),'b_groups':sorted(set(fit_groups)-a_groups),
            'validation_groups':sorted(validation_groups),
            'counts':{'a':len(a),'b':len(b),'validation':len(dev)},
            'original_holdout_used':False,'no_publication_or_campaign':True}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    results = {'plan':plan,'trials':[]}; start=time.perf_counter()
    for seed in plan['seeds']:
        for mode in (plan['arms'] if seed != 29 else plan['arms'][::-1]):
            net=Network(seed,mode,'policy'); rng=np.random.default_rng(seed+3000)
            trial={'seed':seed,'mode':mode,'curve':[]}; began=time.perf_counter()
            for phase in ['A','B']:
                for step in range(plan['steps_per_phase']+1):
                    if step in plan['checkpoints']:
                        point={'phase':phase,'steps_in_phase':step,'optimizer_steps':net.steps,
                               'seconds':time.perf_counter()-began}
                        for label,rows in [('old',a),('new',b),('validation',dev)]:
                            point[label]={'reads':measure(net,rows),
                                          'mean_loss':float(np.mean([net.loss_gradient(r,False) for r in rows]))}
                        trial['curve'].append(point)
                        net.save(args.out/f'{seed}-{mode}-{phase}-{step}.npz')
                        (args.out/f'{seed}-{mode}.json').write_text(json.dumps(trial,indent=2)+'\n')
                        print(json.dumps({'seed':seed,'mode':mode,'phase':phase,'step':step,
                                          'validation_wins':sum(r['win'] for r in point['validation']['reads']),
                                          'validation_loss':point['validation']['mean_loss'],
                                          'old_loss':point['old']['mean_loss'],'new_loss':point['new']['mean_loss']}),flush=True)
                    if step == plan['steps_per_phase']:
                        break
                    net.update([pick(ag,rng), pick(ag if phase=='A' else bg,rng)])
            trial['seconds']=time.perf_counter()-began
            results['trials'].append(trial)
            (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    results['seconds']=time.perf_counter()-start
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'completion.json').write_text(json.dumps({'complete':True,'trials':6,'seconds':results['seconds']},indent=2)+'\n')


if __name__=='__main__':
    main()
