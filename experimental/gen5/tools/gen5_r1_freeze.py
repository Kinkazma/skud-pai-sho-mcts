#!/usr/bin/env python3
"""Freeze rule-only R1 task lots; never load models or change campaign files."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from gen5_r1_corrected_view import materialize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, group):
        self.parent.setdefault(group, group)
        if self.parent[group] != group:
            self.parent[group] = self.find(self.parent[group])
        return self.parent[group]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def audit_groups(indexes):
    groups, orbit_first, splits = Groups(), {}, {}
    exact_splits, symmetry_splits = {}, {}
    for index in indexes:
        for row in lines(index):
            group, held = row['group'], row['held_out']
            if group in splits and splits[group] != held:
                raise ValueError('One source group crosses the original split')
            splits[group] = held
            groups.find(group)
            for name, table in [('exact', exact_splits), ('symmetry', symmetry_splits)]:
                key = row[name]
                table.setdefault(key, set()).add(held)
            key = row['symmetry']
            if key in orbit_first:
                groups.join(group, orbit_first[key])
            else:
                orbit_first[key] = group
    cluster_splits = collections.defaultdict(set)
    for group, split in splits.items():
        cluster_splits[groups.find(group)].add(split)
    quarantined = sorted(g for g in splits if len(cluster_splits[groups.find(g)]) > 1)
    return {
        'clusters': {g: groups.find(g) for g in sorted(splits)},
        'quarantined_groups': quarantined,
        'cross_split_exact_positions': sum(len(v) > 1 for v in exact_splits.values()),
        'cross_split_symmetric_positions': sum(len(v) > 1 for v in symmetry_splits.values()),
        'source_groups': len(splits),
        'statistical_clusters': len(cluster_splits),
    }


def summarize(cases):
    by = collections.defaultdict(list)
    for case in cases:
        by[(case['family'], case['held_out'])].append(case)
    return [{'family': f, 'held_out': h, 'cases': len(xs),
             'source_groups': len({x['group'] for x in xs}),
             'statistical_clusters': len({x['cluster'] for x in xs})}
            for (f, h), xs in sorted(by.items())]


def freeze(args):
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    geometry = [Path(x) for x in args.geometry]
    legacy_index = Path(args.legacy_index)
    census, branches = Path(args.census), Path(args.branches)
    # Frozen cases reference corrected copies, never the historical erroneous
    # counts or an overlay that a future training reader might silently omit.
    census, branches = materialize(census, branches, legacy_index, out / 'corrected-legacy')
    indexes = [legacy_index / 'positions.jsonl'] + [p / 'positions.jsonl' for p in geometry]
    audit = audit_groups(indexes)
    excluded = set(audit['quarantined_groups'])
    cases, dependencies, choice_labels = [], set(indexes), {}

    def add(family, source, line, row, **extra):
        source = Path(source)
        dependencies.add(source)
        identity = f'{family}:{source}:{line}:{row["group"]}'
        cases.append({'id': hashlib.sha256(identity.encode()).hexdigest(),
                      'family': family, 'group': row['group'], 'held_out': row['held_out'],
                      'cluster': audit['clusters'][row['group']],
                      'quarantined': row['group'] in excluded,
                      'reference': {'path': str(source.resolve()), 'zero_based_line_or_index': line},
                      **extra})

    for directory in geometry:
        for i, row in enumerate(lines(directory / 'states.jsonl')):
            if 'acyclic_control' not in row:
                raise ValueError('Current motif lacks a legal acyclic comparison')
            role = 'own' if row['owner'] == row['position']['to_move'] else 'opponent'
            add('current_' + row['family'], directory / 'states.jsonl', i, row,
                perspective_role=role)
        for i, row in enumerate(lines(directory / 'pairs.jsonl')):
            family = 'effect_' + row['family'] + ('_created' if row['new_cycle'] else '_retained')
            add(family, directory / 'pairs.jsonl', i, row,
                origin='legal_alternative' if 'extension_actions' in row else 'human_prefix')
        for item in lines(directory / 'positions.jsonl'):
            if item['file'] == 'pairs.jsonl' and item['role'] == 'root':
                labels = item['all_legal_action_outcomes']
                key = str(directory.resolve()) + ':' + str(item['line'])
                choice_labels[key] = {'group': item['group'], 'held_out': item['held_out'],
                                      'actions': labels,
                                      'winning_actions': [a['action'] for a in labels if a['outcome'] == 'win']}
                if not choice_labels[key]['winning_actions']:
                    raise ValueError('Ring decision root lost all verified wins')
        dependencies.update(directory / x for x in ['verification.json', 'summary.json'])
        dependencies.update(directory.glob('psr/*.psr'))

    roots = json.loads((census / 'roots.json').read_text())
    roots_by_key = {r['key']: r for r in roots}
    legacy_choices = collections.defaultdict(list)
    for i, row in enumerate(lines(branches / 'branches.jsonl')):
        legacy_choices[row['root']].append({'action': row['action'], 'outcome': row['outcome']})
        if row['ending'] == 'exhaustion':
            add('exhaustion_' + row['outcome'], branches / 'branches.jsonl', i, row)
    for key, labels in legacy_choices.items():
        row = roots_by_key[key]
        choice_labels['legacy:' + key] = {'group': row['group'], 'held_out': row['held_out'],
                                        'actions': labels,
                                        'winning_actions': [a['action'] for a in labels if a['outcome'] == 'win']}
    for i, row in enumerate(json.loads((branches / 'contrasts.json').read_text())):
        if row['family'] in ['defence', 'connection_consequence']:
            add(row['family'], branches / 'contrasts.json', i, row)
    dependencies.update([census / 'roots.json', branches / 'branches.jsonl', branches / 'contrasts.json',
                         legacy_index / 'component-corrections.jsonl', legacy_index / 'summary.json'])
    dependencies.update(census.glob('roots/*.psr'))
    sizes = collections.Counter((c['family'], c['cluster']) for c in cases if not c['quarantined'])
    for c in cases:
        c['weight_within_cluster_family'] = 0.0 if c['quarantined'] else 1 / sizes[c['family'], c['cluster']]
    valid = [c for c in cases if not c['quarantined']]
    for row in choice_labels.values():
        row['cluster'] = audit['clusters'][row['group']]
        row['quarantined'] = row['group'] in excluded
        row['eligible_for_immediate_win_metric'] = bool(row['winning_actions']) and not row['quarantined']
    dependencies.update([Path(__file__), Path(__file__).with_name('gen5_r1_corrected_view.py'),
                         out / 'corrected-legacy/manifest.json',
                         Path('docs/research/GEN5_R1_EVALUATION_CONTRACT_V2.md')])
    for name, value in [('lots.json', cases), ('choice-labels.json', choice_labels),
                        ('leakage-audit.json', audit), ('coverage.json', summarize(valid))]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    manifest = {'schema': 'gen5-r1-frozen-lots-v3', 'cases': len(cases), 'usable_cases': len(valid),
                'quarantined_cases': len(cases) - len(valid), 'no_training': True,
                'component_corrections_already_applied': str((out / 'corrected-legacy/manifest.json').resolve()),
                'inputs': {str(p.resolve()): sha(p) for p in sorted(dependencies)},
                'outputs': {p.name: sha(p) for p in sorted(out.glob('*.json'))}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['cases', 'usable_cases', 'quarantined_cases']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['census', 'branches', 'legacy-index', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--geometry', nargs='+', required=True)
    freeze(p.parse_args())
