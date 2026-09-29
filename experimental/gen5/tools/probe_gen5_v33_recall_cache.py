#!/usr/bin/env python3
"""Executable bounded scheduling/cache contracts on frozen, verified V33 evidence."""
import collections as C
import copy
import dataclasses
import gzip
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/gen5-v33-solutions-2026-09-12'
FLOW=ROOT/'benchmarks/results/gen5-v33-flow-audit-2026-09-12'
REVIEW=ROOT/'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
GAMES=ROOT/'training-runs/micro-gen5-ui-20260912-131332-99610d/training/games'
def read(p):return json.loads(Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
panel={p['prefix_sha256']:p for p in read(REVIEW/'panel-results.json')['positions']}
catalogue=[]
for row in read(FLOW/'manifest.json')['positions']:
    if row['kind']!='certificate':continue
    raw=Path(row['path']).read_bytes();assert sha(raw)==row['sha256']
    v=json.loads(raw);key=sha(v['prefix'].encode());p=panel[key]
    catalogue.append(dict(key=key,source=v.get('human_source') or 'guard-'+key,
                          value=int(p['target']),group=row['group']))
assert len(catalogue)==396 and len({x['key'] for x in catalogue})==396
wins=[x for x in catalogue if x['value']==1]

class Coverage:
    """One complete source-interleaved permutation before any item repeats."""
    def __init__(self,rows,seed=20260912):
        self.rows=rows;self.seed=seed;self.epoch=0;self.cursor=0;self.order=self.make_order()
    def make_order(self):
        rng=random.Random(self.seed+self.epoch);groups=C.defaultdict(list)
        for i,row in enumerate(self.rows):groups[row['source']].append(i)
        buckets=list(groups.values());rng.shuffle(buckets)
        for b in buckets:rng.shuffle(b)
        return [b[k] for k in range(max(map(len,buckets))) for b in buckets if k<len(b)]
    def draw(self):
        if self.cursor==len(self.order):self.epoch+=1;self.cursor=0;self.order=self.make_order()
        i=self.order[self.cursor];self.cursor+=1;return self.rows[i]['key']
    def snapshot(self):return dict(seed=self.seed,epoch=self.epoch,cursor=self.cursor,order=self.order)
    @classmethod
    def restore(cls,rows,state):
        c=cls(rows,state['seed']);c.epoch=state['epoch'];c.cursor=state['cursor'];c.order=state['order'];return c

coverage=Coverage(wins)
first=[coverage.draw() for _ in wins]
assert len(set(first))==len(wins)
for _ in range(73):coverage.draw()
restored=Coverage.restore(wins,json.loads(json.dumps(coverage.snapshot())))
assert [coverage.draw() for _ in range(2000)]==[restored.draw() for _ in range(2000)]
# New arrivals do not move old items out of an already-admitted coverage epoch.
current=Coverage(wins);prefix=[current.draw() for _ in range(71)]
pending=dict(key='new-proof',source='new-source',value=1)
tail=[current.draw() for _ in range(len(wins)-71)]
assert set(prefix+tail)=={x['key'] for x in wins}
next_catalogue=wins+[pending];assert pending['key'] in [Coverage(next_catalogue).rows[i]['key'] for i in Coverage(next_catalogue).order]
# Exact existing global quotas with half of each durable lane reserved for coverage.
quotas=C.Counter();cw=Coverage(wins);co=Coverage(catalogue);lane_counts=C.Counter()
for ticket in range(100000):
    lane=ticket%4
    if lane in [0,1]:
        quotas['durable']+=1;name='winning' if lane==0 else 'ordinary';lane_counts[name]+=1
        if lane_counts[name]%2==0:
            (cw if lane==0 else co).draw();quotas[name+'_coverage']+=1
        else:quotas[name+'_priority']+=1
    else:quotas['fresh_or_fifo']+=1
assert quotas['durable']==50000 and quotas['winning_coverage']==12500 and quotas['ordinary_coverage']==12500
# Equal class weight without treating winner-only recall as value supervision.
classes=C.Counter(x['value'] for x in catalogue);least=min(classes.values())
value_mass={z:sum(least/classes[z] for x in catalogue if x['value']==z) for z in classes}
assert max(value_mass.values())-min(value_mass.values())<1e-10
# Preload completion order cannot become learning/sampling order.
ordered=Coverage(wins).order
arrival=ordered.copy();random.Random(29).shuffle(arrival)
ready={i:wins[i]['key'] for i in arrival};assert [ready[i] for i in ordered]==first

@dataclasses.dataclass(frozen=True)
class SearchKey:
    rules:str; model:str; bank:str; prefix:str; budget:int; beta:int
    options:str; root_proof:str; source_exclusion:int=0
def key(row,proof):return SearchKey('skud-pai-sho-gen5-v1',row['model'],'frozen-bank',row['psr'],512,16,'puct/no-noise/no-inherited',proof)
def payload(x):return {k:copy.deepcopy(x.get(k)) for k in ['state','actions','action_features','policy','policy_weight','new_visits','inherited_visits','budget','collector']}|{'search_evidence':{k:copy.deepcopy(x['evidence'].get(k)) for k in ['policy_source','completed_action_values','target_prior','excluded_actions','estimated_value']}}
sample=read(FLOW/'duplicate-verification.json')['sample'];hits=0;misses=0;different_observations=0
for row in sample:
    targets=[]
    for i in row['ids']:
        receipt=read(GAMES/f'game-{i:07}.json');raw=(GAMES/receipt['targets_file']).read_bytes();assert sha(raw)==receipt['targets_sha256']
        targets.append(json.loads(gzip.decompress(raw))[0])
    a,b=targets
    # Retrospective proof state is evidenced by the verified-search replacement.
    # A production adapter must read the actual installed certificate fingerprint.
    pa=a['evidence']['policy_source'] if a['evidence']['policy_source']=='verified-search' else 'none'
    pb=b['evidence']['policy_source'] if b['evidence']['policy_source']=='verified-search' else 'none'
    bank={key(row,pa):payload(a)}
    if key(row,pb) in bank:
        reused=bank[key(row,pb)];assert reused==payload(b);hits+=1
    else:reused=payload(b);misses+=1
    output={'search':reused,'value':b['value'],'observed_value':b['evidence']['observed_value'],'observed_psr':b['evidence']['observed_psr'],'source_run':b['source_run'],'game_id':b['game_id']}
    assert output['value']==b['value'] and output['game_id']==b['game_id']
    different_observations+=a['evidence']['observed_value']!=b['evidence']['observed_value']
    base=key(row,pa)
    for k,v in dict(model='different-model',bank='different-bank',rules='different-rules',prefix='different-prefix',budget=256,beta=8,options='different-options',root_proof='new-certificate',source_exclusion=123).items():
        assert dataclasses.replace(base,**{k:v}) not in bank
assert (hits,misses,different_observations)==(127,1,54)

report=dict(verified_catalogue=len(catalogue),winning_catalogue=len(wins),
            value_classes=dict(C.Counter(x['value'] for x in catalogue)),
            unique_first_epoch=len(set(first)),snapshot_restore_exact=True,
            quota_simulation=dict(quotas),balanced_value_weight_mass=value_mass,
            new_arrivals_do_not_postpone_old_epoch=True,out_of_order_preload_preserves_draw_order=True,
            cache_pairs=len(sample),safe_hits=hits,proof_change_misses=misses,
            distinct_observed_outcomes_preserved=different_observations,key_field_invalidation_tests=len(sample)*9,
            schedule='Keep 50% durable recall, 25% winning policy and 25% other recall. Within each lane, reserve half for complete coverage epochs; keep remaining slots for new/hard/focus items. Balance proof-value gradient by class inside the recall budget; exact mixing strength remains to calibrate.',
            guarantee='Every frozen catalogue entry occurs once per coverage epoch. A different permutation can put two visits up to 2*N-1 coverage slots apart. No fixed wall-time guarantee when catalogue size grows.',
            limitation='Scheduler/cache contract prototypes, not an integrated runtime or a measurement over all 141091 historical proof files; real archive evidence and native search timing are reported separately.')
(OUT/'recall-cache-contracts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
