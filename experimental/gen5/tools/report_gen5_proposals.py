#!/usr/bin/env python3
"""Recompute isolated proposal results; never access mutable training controls."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/gen5-repair-proposals-2026-09-12'


def load(name):
    return json.loads((OUT/name).read_text())


def distribution(row, beta):
    p = np.array(row['prior'])
    if beta == 0:
        return p
    logits = np.log(np.maximum(p, 1e-300)) + beta*np.array(row['successor_values'])
    p = np.exp(logits-logits.max())
    return p/p.sum()


def derivative(rows, beta):
    # Equal total teacher/old weight, matching the proposed 50% recall.
    by_group = collections.defaultdict(list)
    for r in rows:
        if not r['meta']['train']:
            continue
        group = 'old' if r['meta']['cohort']=='old' else 'new'
        by_group[group].append(float(np.dot(distribution(r,beta)-r['target'], r['successor_values'])))
    return sum(np.mean(v) for v in by_group.values())/2


def fit(rows):
    lo,hi = 0.,16.
    if derivative(rows,lo)>=0: return lo
    if derivative(rows,hi)<=0: return hi
    for _ in range(60):
        mid=(lo+hi)/2
        if derivative(rows,mid)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2


def measure(rows, beta):
    result=[]
    for cohort in ['search','opponent','old']:
        for train in [True,False]:
            subset=[r for r in rows if r['meta']['cohort']==cohort and r['meta']['train']==train]
            before=[];after=[];bm=[];am=[]
            for r in subset:
                valid=np.array(r['valid'], dtype=bool)
                p=distribution(r,0);q=distribution(r,beta)
                before.append(bool(valid[p.argmax()]));after.append(bool(valid[q.argmax()]))
                bm.append(float(p[valid].sum()));am.append(float(q[valid].sum()))
            result.append(dict(cohort=cohort,train=train,n=len(subset),before=sum(before),after=sum(after),
                lost=sum(b and not a for b,a in zip(before,after)),gained=sum(a and not b for b,a in zip(before,after)),
                mass_before=float(np.mean(bm)),mass_after=float(np.mean(am))))
    return result


def main():
    old=load('link-features.json');latest=load('latest-link-features.json')
    beta=fit(old['rows'])
    assert beta==16.
    assert [r['meta'] for r in old['rows']]==[r['meta'] for r in latest['rows']]
    train={r['meta']['source'] for r in old['rows'] if r['meta']['train']}
    validation={r['meta']['source'] for r in old['rows'] if not r['meta']['train']}
    assert not train.intersection(validation)
    link=dict(beta=beta,fit_model_identity=old['model_identity'],transferred_without_refit=True,
        latest_model_identity=latest['model_identity'],old=measure(old['rows'],beta),latest=measure(latest['rows'],beta),
        caveat='source-disjoint diagnostic split of campaign-exposed positions; not global held-out strength; successor values are recomputed and are not a learned policy head')
    temporal=[]
    for cohort in ['search','opponent','old']:
        pairs=[(a,b) for a,b in zip(old['rows'],latest['rows']) if a['meta']['cohort']==cohort]
        correct=lambda r:bool(np.array(r['valid'])[distribution(r,beta).argmax()])
        temporal.append(dict(cohort=cohort,n=len(pairs),before=sum(correct(a) for a,b in pairs),
            after=sum(correct(b) for a,b in pairs),lost=sum(correct(a) and not correct(b) for a,b in pairs),
            gained=sum(correct(b) and not correct(a) for a,b in pairs)))
    link['retrospective_between_snapshots']=temporal
    link['retrospective_scope']='same frozen coefficient applied AFTER each archived snapshot; coupling was NOT active during intervening training'
    dist=load('distillation-results.json')
    result=dict(link=link,distillation=[dict(seed=t['seed'],arm=t['arm'],initial=t['trace'][0],final=t['trace'][-1]) for t in dist['trials']],
        imported_teacher_proofs=sum(t['imported'] for t in dist['teacher_searches']),
        rules=load('proposal-rule-results.json'),recall=load('recall-candidate-results.json'))
    if (OUT/'coupled-play/report.json').exists():
        games=load('coupled-play/report.json')['rows']
        result['games']={str(beta):dict(collections.Counter('W' if r['win'] else ('D' if r['outcome']=='Draw' else ('U' if r['outcome']=='Ongoing' else 'L')) for r in games if r['root_value_strength']==beta)) for beta in [0.,16.]}
        replay=[json.loads(x) for x in (OUT/'coupled-play/independent-core-replay.jsonl').read_text().splitlines()]
        assert len(games)==len(replay)==32
        for r,check in zip(games,replay):
            psr=OUT/f"coupled-play/game-{r['id']:03}.psr"
            assert hashlib.sha256(psr.read_bytes()).hexdigest()==r['psr_sha256']
            assert check['path']==str(psr) and check['outcome']==r['outcome']
            assert check['decisions']==r['prefix_decisions']+r['new_decisions']
            assert r['from_zero'] and not r['training_behavior']
        paired=[]
        for pair in range(8):
            for seat in ['Host','Guest']:
                a=next(r for r in games if r['pair']==pair and r['seat']==seat and r['root_value_strength']==0.)
                b=next(r for r in games if r['pair']==pair and r['seat']==seat and r['root_value_strength']==16.)
                label=lambda r:'W' if r['win'] else ('D' if r['outcome']=='Draw' else ('U' if r['outcome']=='Ongoing' else 'L'))
                paired.append(dict(pair=pair,seat=seat,before=label(a),after=label(b)))
        result['game_transitions']=dict(collections.Counter(r['before']+'->'+r['after'] for r in paired))
        # Pair-clustered difference of actual win indicators. U is not labelled
        # a draw/loss: it simply contributes zero to the actual-win indicator.
        delta=np.array([sum((r['after']=='W')-(r['before']=='W') for r in paired if r['pair']==p)/2 for p in range(8)])
        boot=np.random.default_rng(5913).choice(delta,(20000,8),replace=True).mean(axis=1)
        result['game_actual_win_delta']=dict(mean=float(delta.mean()),interval95=np.quantile(boot,[.025,.975]).tolist(),clusters=8,
            caveat='conditional exploratory panel; not an Elo interval or a general strength guarantee')
        result['independent_game_replays']=32
    if (OUT/'shadow-search-16.json').exists():
        native=load('native-search-0.json');zero=load('shadow-search-0.json');coupled=load('shadow-search-16.json')
        assert native==zero and (OUT/'native-search-0.json').read_bytes()==(OUT/'shadow-search-0.json').read_bytes()
        result['search']={'zero_strength_exact':True,'roots':len(zero['rows']),'cohorts':{}}
        for cohort in ['search','opponent',None]:
            pairs=[(a,b) for a,b in zip(zero['rows'],coupled['rows']) if a['cohort']==cohort]
            yes=lambda r:r['known_verified_win'] or r['fresh_verified_win']
            result['search']['cohorts'][str(cohort)]=dict(n=len(pairs),before=sum(yes(a) for a,b in pairs),after=sum(yes(b) for a,b in pairs),lost=sum(yes(a) and not yes(b) for a,b in pairs))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(beta=beta,old=link['old'],latest=link['latest'],imported=result['imported_teacher_proofs'],games=result.get('games'),search=result.get('search')),indent=2))


if __name__=='__main__': main()
