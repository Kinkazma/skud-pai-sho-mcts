#!/usr/bin/env python3
"""Isolated executable specifications for the proposed Gen5 repair, not runtime code."""
import collections
import dataclasses
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/gen5-repair-proposals-2026-09-12'


def evidence_target(observed, estimated, proof=None):
    """Q is retained as a separate estimate; observed outcomes are not overwritten."""
    if proof is not None:
        return proof, 'proof', 1.
    if observed:
        return sum(observed) / len(observed), 'observed', 1.
    if estimated is not None:
        return estimated, 'bootstrap_without_outcome', .25
    return None, 'unlabelled', 0.


def softmax(logits):
    top = max(logits)
    p = [math.exp(x-top) for x in logits]
    z = sum(p)
    return [x/z for x in p]


def q_teacher(prior, completed_q, proofs, strength=1.):
    """Completed Q retains a prior for unknown moves. Only proofs exclude moves."""
    wins = [i for i,p in enumerate(proofs) if p == 1]
    allowed = wins or [i for i,p in enumerate(proofs) if p != -1]
    if not allowed:
        return None
    scores = [math.log(max(prior[i],1e-300)) + (0. if wins else strength*completed_q[i]) for i in allowed]
    p = softmax(scores)
    result = [0.]*len(prior)
    for i,x in zip(allowed,p): result[i] = x
    return result


def winning_projection(prior, verified_winners=None):
    if verified_winners is None:
        return prior.copy()
    mass = sum(p for p,w in zip(prior,verified_winners) if w)
    count = sum(verified_winners)
    assert count
    return [p/mass if w and mass else (1/count if w else 0.) for p,w in zip(prior,verified_winners)]


@dataclasses.dataclass
class Admissions:
    ticket: int = 0
    per_group: list = dataclasses.field(default_factory=lambda: [0]*6)

    def next(self):
        group = 0 if self.ticket % 5 < 4 else (self.ticket//5) % 5 + 1
        ordinal = self.per_group[group]
        opening = ordinal % 20 == (group*3) % 20
        self.per_group[group] += 1
        self.ticket += 1
        return group, opening


def promotion(rows):
    """Proposed frozen lot: one identity/configuration, 50 pairs, no exploration."""
    if len(rows) != 100:
        return False
    keys = ('model','reference','budget','rules','generation')
    if any(len({r[k] for r in rows}) != 1 for k in keys):
        raise ValueError('mixed evaluation identity')
    if any(r['noise'] or r['sampled'] for r in rows):
        raise ValueError('exploratory evaluation')
    pairs = collections.defaultdict(list)
    for r in rows: pairs[r['prefix']].append(r)
    if len(pairs)!=50 or any(sorted(r['seat'] for r in pair)!=['Guest','Host'] for pair in pairs.values()):
        raise ValueError('unpaired or duplicated evaluation starts')
    return sum(r['outcome']=='W' for r in rows)>=60


def retention(entries, evaluated_best=(), keep=64):
    """At most 15 already evaluated champions; preserve recent and time coverage.

    This diagnostic receives compatible champion IDs; it does not rank scores
    across opponents or launch new evaluations. Entries are absolute ordinals.
    """
    entries=sorted(entries)
    if len(entries)<=keep: return entries
    protected=set(entries[-16:]) | {entries[0]} | set(evaluated_best[:15])
    while len(entries)>keep:
        candidates=[i for i in range(1,len(entries)-1) if entries[i] not in protected]
        i=min(candidates,key=lambda i:(entries[i+1]-entries[i-1],entries[i]))
        entries.pop(i)
    return entries


def main():
    r = json.loads((ROOT/'benchmarks/results/gen5-learning-loop-audit-2026-09-12/reanalysis-results.json').read_text())['rows']
    changed = 0
    for row in r:
        if 'original_z' not in row: continue
        observed = [] if row['original_z'] is None else [row['original_z']]
        proof = row['reanalysis_value'] if row['reason']=='search-proven-value' else None
        value,kind,weight = evidence_target(observed,row['reanalysis_value'],proof)
        if row['original_z']==-1 and row['reanalysis_value']>0 and proof is None:
            assert value==-1 and kind=='observed'
            changed += 1
        if proof is not None: assert value==proof
    assert changed==106
    assert evidence_target([-1,1],.9)[0]==0.
    assert evidence_target([-1],.9,1)[0]==1.
    assert evidence_target([],None)[2]==0.

    rng = random.Random(5931)
    worst = 0.
    gradient_error = 0.
    for _ in range(1000):
        size=rng.randrange(2,320)
        logits=[rng.uniform(-5,5) for _ in range(size)]
        prior=softmax(logits)
        q=[rng.uniform(-1,1) for _ in range(size)]
        t=q_teacher(prior,q,[None]*size)
        gain=sum((a-b)*v for a,b,v in zip(t,prior,q))
        assert gain>=-1e-12
        worst=min(worst,gain)
        flat=q_teacher(prior,[.23]*size,[None]*size)
        assert max(abs(a-b) for a,b in zip(flat,prior))<1e-12
        valid=[i%3==0 for i in range(size)]
        conditional=winning_projection(prior,valid)
        assert abs(sum(conditional)-1)<1e-12
        assert all(x==0 for x,w in zip(conditional,valid) if not w)
        assert winning_projection(prior)==prior
        # Check the nontrivial set-loss derivative against central differences.
        for i in [0,size-1]:
            eps=1e-5;plus=logits.copy();minus=logits.copy();plus[i]+=eps;minus[i]-=eps
            loss=lambda v:-math.log(sum(p for p,w in zip(softmax(v),valid) if w))
            numeric=(loss(plus)-loss(minus))/(2*eps)
            analytic=prior[i]-conditional[i]
            gradient_error=max(gradient_error,abs(numeric-analytic))
    assert gradient_error<1e-8
    assert q_teacher([.9,.1],[0.,0.],[None,None])==[.9,.10000000000000003] or max(abs(a-b) for a,b in zip(q_teacher([.9,.1],[0.,0.],[None,None]),[.9,.1]))<1e-14
    assert q_teacher([.5,.5],[-1.,-1.],[-1,-1]) is None

    full=Admissions();expected=[full.next() for _ in range(10000)]
    interrupted=Admissions();actual=[interrupted.next() for _ in range(1337)]
    resumed=Admissions(**json.loads(json.dumps(dataclasses.asdict(interrupted))))
    actual += [resumed.next() for _ in range(10000-1337)]
    assert actual==expected
    groups=collections.Counter(g for g,_ in expected)
    zero=collections.Counter(g for g,z in expected if z)
    assert groups=={0:8000,1:400,2:400,3:400,4:400,5:400}
    assert zero=={0:400,1:20,2:20,3:20,4:20,5:20}
    lot=[dict(model='frozen',reference='ref',budget=8,rules='gen5',generation='3.1',noise=0.,sampled=False,prefix=i//2,seat='Host' if i%2==0 else 'Guest',outcome='W' if i<60 else 'U') for i in range(100)]
    assert promotion(lot)
    low=[dict(r) for r in lot];low[59]['outcome']='D';assert not promotion(low)
    rejected=0
    for key,value in [('model','changed'),('reference','different'),('budget',32),('rules','other'),('generation','3.5'),('noise',.25),('sampled',True),('prefix',1)]:
        invalid=[dict(r) for r in lot];invalid[0][key]=value
        try: promotion(invalid)
        except ValueError: rejected+=1
    assert rejected==8
    anchors=[];saved=None
    for ordinal in range(1061):
        anchors=retention(anchors+[ordinal])
        if ordinal==417: saved=json.loads(json.dumps(anchors))
    assert len(anchors)==64 and anchors[-16:]==list(range(1045,1061))
    assert anchors[0]==0 and max(b-a for a,b in zip(anchors,anchors[1:]))<=64
    for ordinal in range(418,1061): saved=retention(saved+[ordinal])
    assert saved==anchors
    champion=[]
    for ordinal in range(1061): champion=retention(champion+[ordinal],(42,) if ordinal>=42 else ())
    assert 42 in champion and len(champion)==64
    (OUT/'proposal-rule-results.json').write_text(json.dumps(dict(empirical_outcomes_preserved_in_106_unproved_revisions=changed,
        exact_proofs_keep_priority=True,unknown_results_not_fabricated=True,synthetic_q_panels=1000,
        maximum_set_loss_gradient_error=gradient_error,minimum_completed_q_expectation_gain=worst,
        proof_projection_is_identity_without_proof=True,admissions=dict(groups),opening_admissions=dict(zero),
        resume_schedule_exact=True,promotion_rejections=rejected,threshold='60 actual wins / 100 frozen paired games',
        retained_1061_ordinals=anchors,retention_maximum_gap=max(b-a for a,b in zip(anchors,anchors[1:])),
        retention_resume_exact=True,synthetic_evaluated_champion_preserved=True,
        scope='isolated executable specification; no game-strength or full runtime integration claim',production_writes=0),indent=2))
    print('Evidence, Q targets, proof projection, admissions/resume and promotion controls passed.')


if __name__=='__main__': main()
