#!/usr/bin/env python3
"""Materialize audited component corrections in an isolated R1 corpus copy."""
import copy
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corrected(row, patch, root=False):
    """Reject stale/misaddressed patches, preserving every unrelated field."""
    result = copy.deepcopy(row)
    if root:
        if row['key'] != patch['root'] or row['position']['components_host_guest'] != patch['old']:
            raise ValueError('Root correction does not match the audited source')
        result['position']['components_host_guest'] = patch['corrected']
    else:
        if row['root'] != patch['root'] or row['action'] != patch['action']:
            raise ValueError('Branch correction points to a different decision')
        for side in ['before', 'after']:
            field = f'components_{side}_host_guest'
            if row[field] != patch['old_' + side]:
                raise ValueError('Branch correction does not match the audited source')
            result[field] = patch['corrected_' + side]
    return result


def materialize(census, branches, index, out):
    census, branches, index, out = map(Path, [census, branches, index, out])
    audit = json.loads((index / 'summary.json').read_text())
    sources = [census / 'roots.json', branches / 'branches.jsonl',
               branches / 'contrasts.json', index / 'component-corrections.jsonl',
               index / 'summary.json'] + sorted(census.glob('roots/*.psr'))
    inputs = {str(p.resolve()): sha(p) for p in sources}
    for path, key in [(census / 'roots.json', 'roots_sha256'),
                      (branches / 'branches.jsonl', 'branches_sha256')]:
        if inputs[str(path.resolve())] != audit[key]:
            raise ValueError('Corpus changed since the rule-only correction audit')
    patches = {'roots.json': {}, 'branches.jsonl': {}}
    for line in (index / 'component-corrections.jsonl').read_text().splitlines():
        patch = json.loads(line)
        target = patches[patch['file']]
        if patch['line'] in target or patch['line'] < 0:
            raise ValueError('Duplicate or invalid correction index')
        target[patch['line']] = patch
    for name, key in [('roots.json', 'corrected_root_components'),
                      ('branches.jsonl', 'corrected_branch_components')]:
        if len(patches[name]) != audit[key]:
            raise ValueError('Incomplete correction overlay')
    roots = json.loads((census / 'roots.json').read_text())
    if len(roots) != audit['roots']:
        raise ValueError('Root census changed')
    for i, patch in patches['roots.json'].items():
        roots[i] = corrected(roots[i], patch, root=True)
    out.mkdir(exist_ok=False)
    dest_census, dest_branches = out / 'census', out / 'branches'
    dest_census.mkdir()
    dest_branches.mkdir()
    (dest_census / 'roots.json').write_text(json.dumps(roots, indent=2) + '\n')
    shutil.copytree(census / 'roots', dest_census / 'roots')
    shutil.copyfile(branches / 'contrasts.json', dest_branches / 'contrasts.json')
    count, applied = 0, 0
    with (branches / 'branches.jsonl').open() as source, (dest_branches / 'branches.jsonl').open('w') as dest:
        for i, line in enumerate(source):
            if i in patches['branches.jsonl']:
                line = json.dumps(corrected(json.loads(line), patches['branches.jsonl'][i])) + '\n'
                applied += 1
            dest.write(line)
            count += 1
    if count != audit['branches'] or applied != len(patches['branches.jsonl']):
        raise ValueError('Branch correction coverage mismatch')
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Source changed during materialization')
    manifest = {'schema': 'gen5-r1-corrected-components-v2', 'no_training': True,
                'roots': len(roots), 'branches': count,
                'corrected_roots': len(patches['roots.json']), 'corrected_branches': applied,
                'inputs': inputs, 'tool_sha256': sha(__file__),
                'outputs': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dest_census, dest_branches
