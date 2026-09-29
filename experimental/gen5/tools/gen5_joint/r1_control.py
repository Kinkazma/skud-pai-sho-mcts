"""Post-trial R1 control, including complete positive/negative threat censuses.

No fitting or threshold selection. These R1 controls were consulted in the older
R2 comparison; they are not advertised as previously unseen scientific evidence.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .network import JointNetwork, prepare
from .trial import measure


def ranking(scores, truth):
    """Exact threshold ties; descriptive action-level AP/AUC, not independent trials."""
    scores=np.concatenate(scores);truth=np.concatenate(truth).astype(int)
    _,inverse=np.unique(scores,return_inverse=True)
    positive=np.bincount(inverse,weights=truth);negative=np.bincount(inverse,weights=1-truth)
    p=float(positive.sum());n=float(negative.sum())
    auc=float((positive*(np.cumsum(negative)-.5*negative)).sum()/(p*n)) if p and n else None
    tp=np.cumsum(positive[::-1]);fp=np.cumsum(negative[::-1])
    ap=float((positive[::-1]*tp/np.maximum(1,tp+fp)).sum()/p) if p else None
    return {'average_precision':ap,'roc_auc':auc,'positive_prevalence':p/max(1,p+n),'positive':int(p),'negative':int(n)}


def defence_choices(rows, predictions):
    result={'safe':0,'unsafe':0,'unknown':0,'eligible_roots':0}
    for r,prediction in zip(rows,predictions):
        known=r['known'][:,5];danger=r['events'][:,5]>0
        if r['winning'].any() or not (known&danger).any() or not (known&~danger).any():continue
        result['eligible_roots']+=1;best=int(prediction.argmax())
        result['unknown' if not known[best] else 'unsafe' if danger[best] else 'safe']+=1
    return result


def main(root):
    corpus=root/'r1-control-corpus'
    summary=json.loads((corpus/'summary.json').read_text())
    assert hashlib.sha256((corpus/'rows.jsonl').read_bytes()).hexdigest()==summary['rows_sha256']
    rows=[prepare(json.loads(line)) for line in (corpus/'rows.jsonl').open()]
    plan=json.loads((root/'plan.json').read_text())
    trained={s['group'] for s in plan['sources'] if s['split']=='train'}
    assert not trained & {r['group'] for r in rows}
    results=[]
    for seed in plan['seeds']:
        for arm in plan['arms']:
            mode,kind=arm.split('-');net=JointNetwork(seed,mode,kind=='joint')
            path=root/'learning'/f'{seed}-{arm}-B-{plan["steps_per_phase"]}.npz'
            with np.load(path) as data:
                net.w={k:data[k].copy() for k in net.w}
                net.steps=int(data['steps'])
            readings=measure(net,rows,auxiliary=True)
            counts={k:np.sum([r['events'][k] for r in readings],axis=0).tolist() for k in ['tp','fp','tn','fn']}
            predictions=[]
            score_bins=[[] for _ in range(10)];truth_bins=[[] for _ in range(10)]
            for r in rows:
                prediction,cache=net.forward(r)
                predictions.append(prediction)
                for j in range(10):
                    known_head=r['known'][:,j]
                    score_bins[j].append(cache[-1][known_head,11+j]);truth_bins[j].append(r['events'][known_head,j])
            results.append({'seed':seed,'arm':arm,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'wins':sum(r['win'] for r in readings),'events':counts,'defence':defence_choices(rows,predictions),'readings':readings,
                'event_ranking':[ranking(s,t) for s,t in zip(score_bins,truth_bins)]})
    result={'scope':__doc__,'roots':len(rows),'groups':len({r['group'] for r in rows}),
        'winning_roots':sum(bool(r['winning'].any()) for r in rows),
        'baseline_wins':sum(bool(r['winning'][r['prior'].argmax()]) for r in rows),
        'baseline_defence':defence_choices(rows,[r['prior'] for r in rows]),'trials':results}
    (root/'r1-control-readings.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in results:print(r['arm'],r['seed'],r['wins'],r['defence'],{k:r['events'][k][5] for k in r['events']})


if __name__=='__main__':main(Path(sys.argv[1]))
