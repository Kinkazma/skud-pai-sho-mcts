"""Prepare the R2 experiment only from verified frozen R1 references."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(frozen, model, out):
    frozen, model, out = map(Path, [frozen, model, out])
    manifest = json.loads((frozen / 'manifest.json').read_text())
    if manifest['schema'] != 'gen5-r1-frozen-lots-v3':
        raise ValueError('Corrected, materialized V3 lots are required')
    for path, digest in manifest['inputs'].items():
        if sha(path) != digest:
            raise ValueError('Frozen R1 dependency changed: ' + path)
    for path, digest in manifest['outputs'].items():
        if sha(frozen / path) != digest:
            raise ValueError('Frozen R1 output changed: ' + path)
    out.mkdir(exist_ok=False)
    shutil.copyfile(model, out / 'actor.json')
    view = frozen / 'corrected-legacy'
    roots = json.loads((view / 'census/roots.json').read_text())
    roots_by_key = {r['key']: r for r in roots}
    choices = json.loads((frozen / 'choice-labels.json').read_text())
    rows, metadata = [], {}
    for key, info in choices.items():
        if info['quarantined'] or not info['winning_actions']:
            continue
        if key.startswith('legacy:'):
            row = roots_by_key[key.split(':', 1)[1]]
            psr = view / 'census' / row['psr']
            digest = row['sha256']
        else:
            directory, line = key.rsplit(':', 1)
            row = json.loads((Path(directory) / 'pairs.jsonl').read_text().splitlines()[int(line)])
            psr = Path(directory) / row['root']['path']
            digest = row['root']['sha256']
        rows.append({'task': 'policy', 'reference': key, 'psr': str(psr.resolve()),
                     'sha256': digest, 'group': info['group'], 'held_out': info['held_out'],
                     'winning_actions': info['winning_actions']})
    seen = set()
    for case in json.loads((frozen / 'lots.json').read_text()):
        if case['quarantined'] or not case['family'].startswith('current_'):
            continue
        ref = case['reference']
        path = Path(ref['path'])
        index = ref['zero_based_line_or_index']
        identity = (str(path), index)
        if identity in seen:
            continue
        seen.add(identity)
        row = json.loads(path.read_text().splitlines()[index])
        slot = (0 if row['family'] == 'off_centre' else 2) + int(row['owner'] != row['position']['to_move'])
        for role, target in [('root', 1), ('acyclic_control', 0)]:
            record = row[role]
            rows.append({'task': 'motif', 'reference': f'{path}:{index}:{role}',
                         'psr': str((path.parent / record['path']).resolve()),
                         'sha256': record['sha256'], 'group': row['group'],
                         'held_out': row['held_out'], 'slot': slot, 'target': target})
    for r in rows:
        if sha(r['psr']) != r['sha256']:
            raise ValueError('Replay changed: ' + r['psr'])
        if r['group'] in metadata and metadata[r['group']] != r['held_out']:
            raise ValueError('Group crosses the original split')
        metadata[r['group']] = r['held_out']
    plan = {'schema': 'gen5-r2-native-plan-v1', 'actor': str((out / 'actor.json').resolve()),
            'actor_sha256': sha(out / 'actor.json'), 'original_actor': str(model.resolve()),
            'frozen_manifest': str((frozen / 'manifest.json').resolve()),
            'frozen_manifest_sha256': sha(frozen / 'manifest.json'), 'rows': rows,
            'seeds': [17, 29, 43], 'steps_per_phase': 1024, 'phase_B_recall': 0.5,
            'activation': False, 'no_new_search': True}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'references': len(rows), 'policy': sum(r['task'] == 'policy' for r in rows),
                      'motif': sum(r['task'] == 'motif' for r in rows)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frozen'); p.add_argument('actor'); p.add_argument('out')
    a = p.parse_args()
    prepare(a.frozen, a.actor, a.out)
