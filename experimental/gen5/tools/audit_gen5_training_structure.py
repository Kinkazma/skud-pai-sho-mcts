#!/usr/bin/env python3
"""Read-only receipt/FIFO/durable sampling audit for a stopped Gen5 campaign."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    p = args.campaign.resolve()
    root = p / 'training'
    plan = json.loads((root / 'plan.json').read_text())
    options = plan['options']
    receipts = []
    for path in sorted((root / 'games').glob('game-*.json')):
        if '.targets.' not in path.name:
            receipts.append((path, json.loads(path.read_text())))
    groups = defaultdict(list)
    for _, r in receipts:
        groups[r['lane']].append(r)
    stats = {}
    for lane, rows in groups.items():
        batches = Counter()
        for r in rows:
            used = r['fresh_used'] + r['replay_used'] + r['human_used']
            # All fully consumed game lists have precisely this tail. Censored
            # partial games are excluded from tail-size inference.
            if r.get('fully_learned') and used:
                batches[64] += used // 64
                if used % 64:
                    batches[used % 64] += 1
        stats[lane] = dict(
            receipts=len(rows),
            fresh=sum(r['fresh_used'] for r in rows),
            replay=sum(r['replay_used'] for r in rows),
            human=sum(r['human_used'] for r in rows),
            durable=sum(r['durable_draws'] for r in rows),
            optimizer_steps=sum(r['learned_batches'] for r in rows),
            inferred_complete_batch_sizes=dict(sorted(batches.items())),
            zero_durable_receipts=sum(r['durable_draws'] == 0 for r in rows),
            terminations=dict(Counter(r['termination'] for r in rows)),
        )
    fifo = json.loads((root / 'replay-final.index.json').read_text())
    origins = Counter()
    for r in fifo['rows']:
        origins[f"{Path(r['source']['path']).parents[2].name}/{r['lane']}"] += len(r['indices'])
    # Deterministic stratified sample of actual dense target files. No historical
    # revision is substituted: these are precisely the immutable FIFO labels.
    sampled = []
    for lane, maximum in [('Historical', 100), ('Reanalysis', 300)]:
        rows = [(path, r) for path, r in receipts if r['lane'] == lane and r['eligible_examples']]
        ids = sorted(set(i * (len(rows)-1) // max(1, min(maximum,len(rows))-1)
                         for i in range(min(maximum,len(rows)))))
        sampled.extend(rows[i] for i in ids)
    reasons = Counter()
    proof_count = policy_count = negative_policy = example_count = 0
    identities = []
    identical = defaultdict(list)
    for path, r in sampled:
        target = path.parent / r['targets_file']
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != r['targets_sha256']:
            raise ValueError(f'changed target {target}')
        examples = json.loads(gzip.decompress(data))
        if len(examples) != r['eligible_examples']:
            raise ValueError('target count mismatch')
        identities.append(dict(path=str(target), sha256=r['targets_sha256'], examples=len(examples)))
        for ex in examples:
            example_count += 1
            reasons[ex['reason']] += 1
            proof_count += ex.get('tactical', {}).get('root_value') is not None if ex.get('tactical') else 0
            policy_count += ex['policy_weight'] > 0
            negative_policy += ex['value'] < 0 and ex['policy_weight'] > 0
            key = json.dumps(ex['state'], separators=(',', ':'))
            identical[key].append((ex['value'], ex['reason']))
    collisions = [v for v in identical.values() if len(v)>1]
    archive = Path(options['case_curriculum']['archive'])
    bundles = sorted(archive.glob('*.json.gz'))
    ids = sorted(set(i*(len(bundles)-1)//min(599,len(bundles)-1) for i in range(min(600,len(bundles)))))
    durable_rows = []
    source_exclusion = []
    artifact = json.loads((root/'model.json').read_text())
    sources_path = Path(artifact['sequence_memory']['path']).parent/'sources.json'
    bank_sources = {s['sha256']:s for s in json.loads(sources_path.read_text())}
    for i in ids:
        path = bundles[i]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.name.removesuffix('.json.gz'):
            raise ValueError('durable hash mismatch')
        b = json.loads(gzip.decompress(data))
        if hashlib.sha256(b['psr'].encode()).hexdigest() != b['psr_sha256']:
            raise ValueError('durable PSR hash mismatch')
        durable_rows.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),
                                 lessons=len(b['lessons']), kind=b['case']['kind'],
                                 proofs=len(b['proofs'])))
        source = bank_sources.get(b['psr_sha256'])
        if source and not source['human']:
            actual = f"{b['source']}/{b['game_id']}"
            source_exclusion.append(dict(bundle=str(path),psr_sha256=b['psr_sha256'],
                bank_source=source['source'],durable_source=actual,
                same=hashlib.sha256(actual.encode()).digest()[:8] ==
                     hashlib.sha256(source['source'].encode()).digest()[:8]))
    ladder = json.loads((root/'progress.json').read_text())['legacy_ladder']
    report = dict(
        campaign=str(p), options=options,
        input_hashes={str(f):digest(f) for f in [root/'model.json',root/'replay-final.index.json',root/'plan.json',p/'status.json']},
        receipt_count=len(receipts), lanes=stats,
        fresh_coverage=sum(r['policy_coverage_sum'] for _,r in receipts)/sum(r['policy_searches'] for _,r in receipts),
        current_human_sources=len({r['case']['human_source'] for _,r in receipts}),
        fifo_positions=sum(origins.values()),fifo_sources=len(fifo['rows']),fifo_origins=dict(origins),
        ladder_stage=ladder['stage'],ladder_batches=ladder['batches'],
        source_exclusion=source_exclusion,
        dense_sample=dict(files=identities,positions=example_count,reasons=dict(reasons),
                          proofs=proof_count,policy_examples=policy_count,negative_value_with_policy=negative_policy,
                          exact_input_repeated_groups=len(collisions),
                          exact_input_opposite_signed_target_groups=sum(min(t[0] for t in v)<0<max(t[0] for t in v) for v in collisions)),
        durable_sample=dict(population=len(bundles),rows=durable_rows,
                            kinds=dict(Counter(r['kind'] for r in durable_rows)),
                            singleton_bundles=sum(r['lessons']==1 for r in durable_rows),
                            median_lessons=statistics.median(r['lessons'] for r in durable_rows),
                            proofs=sum(r['proofs'] for r in durable_rows)),
    )
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    args.output.with_name('source-exclusion.json').write_text(json.dumps(source_exclusion,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['receipt_count','lanes','fifo_positions','fifo_origins','fresh_coverage']},indent=2))


if __name__ == '__main__':
    main()
