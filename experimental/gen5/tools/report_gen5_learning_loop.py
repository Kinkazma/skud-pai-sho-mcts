#!/usr/bin/env python3
"""Summarize frozen evidence; never reads/writes the running model or controls."""
import collections
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/gen5-learning-loop-audit-2026-09-12'


def read(name):
    return json.loads((OUT / name).read_text())


def softmax(x):
    top = max(x)
    p = [math.exp(v - top) for v in x]
    z = sum(p)
    return [v / z for v in p]


def linked(row, beta):
    return softmax([math.log(max(p, 1e-300)) + beta * q
                    for p, q in zip(row['prior'], row['successor_values'])])


def link_metrics(rows, beta):
    masses, loss, correct = [], [], 0
    for row in rows:
        p = linked(row, beta)
        certified = row['certified']
        n = sum(certified)
        masses.append(sum(v for v, yes in zip(p, certified) if yes))
        loss.append(-sum(math.log(max(v, 1e-300)) for v, yes in zip(p, certified) if yes) / n)
        correct += certified[max(range(len(p)), key=lambda i: p[i])]
    return dict(n=len(rows), certified_mass=st.mean(masses), cross_entropy=st.mean(loss), certified_top=correct)


def main():
    roots = read('root-results.json')['rows']
    summary = {'roots': {}}
    for kind in ['case', 'certificate', 'immediate']:
        group = [r for r in roots if r['kind'] == kind]
        arms = {}
        for name in [a['arm'] for a in group[0]['searches']]:
            chosen = [next(a for a in row['searches'] if a['arm'] == name) for row in group]
            arms[name] = dict(n=len(group), certified_original=sum(a['certified'] for a in chosen),
                verified_wins=sum(a['certified'] or a.get('proved_actions', [None] * row['legal'])[a['selected']] == 1
                                  for row, a in zip(group, chosen)),
                median_evaluations=st.median(a.get('evaluated_actions', a.get('inference_evaluations', 0)) for a in chosen),
                median_visited=st.median(a['visited'] for a in chosen))
        picks = {}
        for name in ['policy_pick', 'value_pick']:
            picks[name] = dict(original_certified=sum(r['certified'][r[name]] for r in group),
                verified_wins=sum(r['certified'][r[name]] or any(a.get('proved_actions', [None] * r['legal'])[r[name]] == 1
                                                               for a in r['searches']) for r in group))
        summary['roots'][kind] = dict(n=len(group), median_legal=st.median(r['legal'] for r in group), picks=picks, arms=arms)

    # A one-coefficient value-to-policy bridge, not a new neural architecture.
    # Fit only on source hashes outside the validation partition; no tuning on validation.
    panel = [r for r in roots if r['kind'] == 'certificate']
    train = [r for r in panel if hashlib.sha256(r['source'].encode()).digest()[0] % 3 != 0]
    valid = [r for r in panel if r not in train]
    assert set(r['source'] for r in train).isdisjoint(r['source'] for r in valid)
    def derivative(beta):
        total = 0.
        for r in train:
            p = linked(r, beta)
            total += sum(v*q for v,q in zip(p,r['successor_values']))
            total -= sum(q for q,w in zip(r['successor_values'],r['certified']) if w)/sum(r['certified'])
        return total / len(train)
    low, high = 0., 16.
    if derivative(low) >= 0:
        beta = low
    elif derivative(high) <= 0:
        beta = high
    else:
        for _ in range(70):
            mid = (low + high) / 2
            if derivative(mid) > 0: high = mid
            else: low = mid
        beta = (low + high) / 2
    summary['value_policy_bridge'] = dict(beta=beta, bounds=[0,16],
        train_sources=[r['source'] for r in train], validation_sources=[r['source'] for r in valid],
        train_before=link_metrics(train,0), train_after=link_metrics(train,beta),
        validation_before=link_metrics(valid,0), validation_after=link_metrics(valid,beta),
        immediate_before=link_metrics([r for r in roots if r['kind']=='immediate'],0),
        immediate_after=link_metrics([r for r in roots if r['kind']=='immediate'],beta),
        scope='one scalar fitted offline; validation source-disjoint from this fit but all archival positions campaign-exposed; no model saved or activation')

    play = read('paired-play/report.json')['rows']
    verified = [json.loads(l) for l in (OUT/'paired-play/independent-core-replay.jsonl').read_text().splitlines()]
    assert len(verified) == len(play) == 32
    for r,v in zip(play,verified):
        assert v['outcome'] == r['outcome']
        assert v['decisions'] == r['new_decisions'] + r['prefix_decisions']
        assert hashlib.sha256((OUT/f"paired-play/game-{r['id']:03}.psr").read_bytes()).hexdigest() == r['psr_sha256']
    summary['paired_play'] = {}
    for training in [True,False]:
        group = [r for r in play if r['training_behavior'] == training]
        counts = collections.Counter('U' if r['outcome']=='Ongoing' else 'D' if r['outcome']=='Draw' else 'W' if r['win'] else 'L' for r in group)
        summary['paired_play'][str(training)] = dict(counts=counts, n=len(group),
            candidate_decisions=sum(len(r['decisions']) for r in group),
            changed_by_sampling=sum(d['different'] for r in group for d in r['decisions']),
            new_decisions=sum(r['new_decisions'] for r in group))
    summary['paired_play']['independently_replayed'] = len(verified)

    feedback = read('reanalysis-results.json')['rows']
    summary['reanalysis'] = dict(sample=len(feedback), origin_linked=sum('original_z' in r for r in feedback), by_outcome={})
    for z in [-1,0,1,None]:
        group = [r for r in feedback if 'original_z' in r and r['original_z'] == z]
        summary['reanalysis']['by_outcome'][str(z)] = dict(n=len(group), positive=sum(r['reanalysis_value']>0 for r in group),
            positive_unproved=sum(r['reanalysis_value']>0 and r['reason']=='fresh-reanalysis-q' for r in group),
            positive_proved=sum(r['reanalysis_value']>0 and r['reason']=='search-proven-value' for r in group))
    flow = read('reference-flow-results.json')['rows']
    summary['reference_flow'] = {k:sum(r[k] for r in flow) for k in [
        'candidate_decisions','opponent_decisions','candidate_under_sampling_rule','sampling_beyond_absolute40',
        'opponent_immediate_winning_decisions','candidate_immediate_winning_decisions','fresh_targets']}
    summary['reference_flow']['games'] = len(flow)
    summary['learning_flow'] = read('flow.json')
    journal = ROOT/'benchmarks/results/gen5-campaign-review-2026-09-11/receipts-phase-2.jsonl'
    games = [r for line in journal.read_text().splitlines() if not (r:=json.loads(line))['reanalysis']]
    starts = [r for r in games if r['prefix'] == 0]
    summary['starts'] = dict(continuations=len(games), from_zero=len(starts), unique_zero_prefixes=len({r['case'] for r in starts}),
        lanes=dict(collections.Counter(r['lane'] for r in starts)), references_at_most40_decisions=sum(r['decisions']<=40 for r in games if r['lane']=='Historical'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary['value_policy_bridge'].items() if not k.endswith('_sources')},indent=2))
    print(json.dumps(summary['reference_flow'],indent=2))


if __name__ == '__main__':
    main()
