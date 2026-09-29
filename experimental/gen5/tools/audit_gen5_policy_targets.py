"""Read-only systematic snapshot of retained V5 targets; no campaign mutation."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def audit(campaign, output, maximum=1024):
    campaign, output = Path(campaign), Path(output)
    output.mkdir(exist_ok=True)
    files = sorted((campaign / 'training/games').glob('*.targets.json.gz'))
    selected = [files[i * len(files) // min(maximum, len(files))]
                for i in range(min(maximum, len(files)))]
    groups = collections.defaultdict(list)
    manifest, missing = [], []
    for path in selected:
        try:
            raw = path.read_bytes()
            receipt = json.loads(path.with_name(path.name.replace('.targets.json.gz', '.json')).read_text())
        except FileNotFoundError:
            missing.append(str(path))
            continue
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == receipt['targets_sha256']
        examples = json.loads(gzip.decompress(raw))
        assert len(examples) == receipt['eligible_examples']
        manifest.append({'path': str(path), 'sha256': digest, 'examples': len(examples)})
        for x in examples:
            p = x['policy']
            if x['policy_weight'] <= 0 or not p:
                continue
            assert len(x['state']) == 417 and abs(sum(p) - 1) < 1e-8
            support = sum(t > 0 for t in p)
            entropy = -sum(t * math.log(t) for t in p if t > 0)
            new = x['new_visits']
            tactical = x.get('tactical') or {}
            proof = tactical.get('root_value') is not None
            row = dict(legal=len(p), visited_new=sum(n > 0 for n in new), support=support,
                       max_target=max(p), effective_actions=math.exp(entropy),
                       entropy_support=entropy / math.log(support) if support > 1 else 0.,
                       inherited=x['inherited_visits'], pruned=sum(x.get('policy_pruned_visits', [])),
                       any_action_proof=any(v is not None for v in tactical.get('action_values', [])))
            row['visited_fraction'] = row['visited_new'] / row['legal']
            for lane in ('all', receipt['lane']):
                groups[f"{x['budget']}:{'root-proven' if proof else 'root-unproven'}:{lane}"].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        result[name] = {'examples': len(rows), 'medians': {
            key: statistics.median(r[key] for r in rows) for key in rows[0]},
            'near_uniform_fraction': sum(r['entropy_support'] > .98 for r in rows) / len(rows),
            'targets_with_pruning': sum(r['pruned'] > 0 for r in rows),
            'zero_target_legal_fraction_mean': statistics.mean(1 - r['support'] / r['legal'] for r in rows)}
    data = dict(available_at_listing=len(files), sampled_files=len(manifest), missing_during_read=missing,
                examples=sum(r['examples'] for r in manifest), groups=result, manifest=manifest,
                scope='Systematic retained-file sample; not all campaign targets; frozen behavior priors absent.')
    (output / 'target-statistics.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'manifest'}, indent=2))


if __name__ == '__main__':
    audit(*sys.argv[1:])
