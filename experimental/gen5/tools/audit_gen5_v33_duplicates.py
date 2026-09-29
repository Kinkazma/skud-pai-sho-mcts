#!/usr/bin/env python3
"""Read-only census plus a bounded, hash-checked sample of repeated reanalyses."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
OUT = ROOT / 'benchmarks/results/gen5-v33-flow-audit-2026-09-12'
GAMES = ROOT / 'training-runs/micro-gen5-ui-20260912-131332-99610d/training/games'
rows = [json.loads(s) for s in (BASE / 'receipts.jsonl').open()]
groups = collections.defaultdict(list)
for row in rows:
    if row['reanalysis']:
        groups[(row['collector'], row['psr_sha256'])].append(row)
duplicates = [rs for rs in groups.values() if len(rs) > 1]
available = []
for rs in duplicates:
    have = [r for r in rs if (GAMES / r['targets_file']).exists()]
    if len(have) > 1:
        available.append(have)
selected = [available[i * (len(available)-1)//127] for i in range(128)]
result = []
hashes = {}
for rs in selected:
    pair = [rs[0], rs[-1]]
    targets = []
    for row in pair:
        p = GAMES / row['targets_file']
        raw = p.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        assert h == row['targets_sha256']
        hashes[str(p)] = h
        psr = GAMES / f"game-{row['id']:07}.psr"
        assert hashlib.sha256(psr.read_bytes()).hexdigest() == row['psr_sha256']
        hashes[str(psr)] = row['psr_sha256']
        x = json.loads(gzip.decompress(raw))
        assert len(x) == 1
        targets.append(x[0])
    a, b = targets
    fields = ['rules', 'collector', 'budget', 'inherited_visits', 'new_visits',
              'decision', 'actions', 'state', 'action_features', 'policy',
              'value', 'policy_weight']
    same = {k: a.get(k) == b.get(k) for k in fields}
    for k in ['policy_source', 'completed_action_values', 'target_prior',
              'excluded_actions', 'estimated_value', 'value_weight',
              'observed_value', 'observed_psr']:
        same['evidence.'+k] = a['evidence'].get(k) == b['evidence'].get(k)
    result.append(dict(ids=[r['id'] for r in pair], model=pair[0]['collector'],
                       psr=pair[0]['psr_sha256'], equal=same))
extras = [r for rs in duplicates for r in rs[1:]]
report = dict(
    reanalyses=sum(map(len, groups.values())), unique_model_psr=len(groups),
    repeated_groups=len(duplicates), repeated_excess=len(extras),
    max_repetitions=max(map(len, groups.values())),
    duplicate_excess_summed_actor_search_seconds=sum(sum(r['search_seconds']) for r in extras),
    duplicate_excess_fresh_positions=sum(r['fresh_used'] for r in extras),
    retained_groups_available=len(available), sample_pairs=len(result),
    sample_equal_counts={k: sum(x['equal'][k] for x in result) for k in result[0]['equal']},
    sample=result, input_hashes=hashes,
    limitation='Systematic 128-pair sample among retained dense groups, not all historical repetitions. Identical search does not imply identical empirical outcome/provenance, and is not a wall-clock saving estimate.')
(OUT / 'duplicate-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['sample','input_hashes']}, indent=2))
