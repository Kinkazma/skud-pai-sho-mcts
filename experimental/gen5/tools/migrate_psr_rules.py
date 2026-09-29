#!/usr/bin/env python3
"""Revalidate a PSR corpus under current rules into a new, auditable directory.

No search or training. Original files, external outcomes and split identities
are preserved in receipts; Q/visit targets are never copied.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_external(value, identity):
    if (value.get('schema') != 'paisho-external-outcome-v1'
            or value.get('record_sha256') != identity
            or value.get('kind') != 'site_resignation'
            or value.get('outcome') not in ('H', 'G')
            or not value.get('source_records')):
        raise ValueError('invalid external resignation or source record hash')
    for source in value['source_records']:
        if (not isinstance(source.get('game_id'), int) or source['game_id'] <= 0
                or not valid_sha(source.get('original_sha256'))
                or not valid_sha(source.get('metadata_sha256'))):
            raise ValueError('invalid external resignation provenance')


def migrate(source, output, converter):
    source, output, converter = source.resolve(), output.resolve(), converter.resolve()
    files = sorted(source.rglob('*.psr')) if source.is_dir() else [source]
    if not files or any(not p.is_file() for p in files):
        raise ValueError('source must contain at least one PSR file')
    # Do not admit a destination inside the source scan: future imports must not
    # accidentally rediscover this conversion as additional independent data.
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError('output must be outside the source corpus')
    output.mkdir(parents=True, exist_ok=False)
    records_dir = output / 'records'
    records_dir.mkdir()
    save(output / 'plan.json', dict(schema='paisho-rules-corpus-migration-v1',
         source=str(source), converter=str(converter), converter_sha256=sha(converter.read_bytes()),
         files=len(files), search_targets='not transferred', originals='unchanged',
         split='original canonical identity retained', destination=str(output)))
    rows, groups = [], {}
    for index, path in enumerate(files):
        row = dict(original_path=str(path), original_file_sha256=sha(path.read_bytes()))
        try:
            with tempfile.TemporaryDirectory(prefix='paisho-rules-') as tmp:
                target, canonical = Path(tmp) / 'target.psr', Path(tmp) / 'source.psr'
                result = subprocess.run([str(converter), str(path), str(target), str(canonical)],
                                        text=True, capture_output=True, timeout=120)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                facts = json.loads(result.stdout)
                data = target.read_bytes()
                source_hash, target_hash = sha(canonical.read_bytes()), sha(data)
            row.update(facts, source_record_sha256=source_hash, target_record_sha256=target_hash)
            split_hash = source_hash
            previous_path = path.with_suffix('.rules-migration.json')
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                if (previous.get('schema') != 'paisho-rules-migration-v1'
                        or previous.get('target_record_sha256') != source_hash
                        or previous.get('target_rules') != facts['source_rules']
                        or not valid_sha(previous.get('split_identity_sha256'))):
                    raise ValueError('invalid prior migration receipt')
                split_hash = previous['split_identity_sha256']
                row['previous_receipt_sha256'] = sha(previous_path.read_bytes())
            row['split_identity_sha256'] = split_hash
            external_path = path.with_suffix('.outcome.json')
            external = None
            if external_path.exists():
                original_external = json.loads(external_path.read_text())
                validate_external(original_external, source_hash)
                if facts['source_outcome'] != 'ongoing':
                    raise ValueError('source external resignation requires an ongoing PSR')
                row['original_external_outcome'] = original_external
                row['original_external_sha256'] = sha(external_path.read_bytes())
                if facts['target_outcome'] == 'ongoing':
                    external = dict(original_external, record_sha256=target_hash)
                    row['external_policy'] = 'retained_on_revalidated_ongoing_record'
                else:
                    row['external_policy'] = 'superseded_by_earlier_rules_terminal'
            row['status'] = 'converted'
            row['record_path'] = f'records/{index:06d}-{target_hash}.psr'
            groups.setdefault(target_hash, []).append((row, data, external))
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            row.update(status='excluded', reason=str(error))
        rows.append(row)
    for group in groups.values():
        # Different V1 suffixes can collapse to one V2 terminal prefix. Do not
        # silently let a train/validation collision enter the next human fit.
        splits = {r['split_identity_sha256'] for r, _, _ in group}
        targets = {e['outcome'] if e else r['target_outcome'] for r, _, e in group
                   if e or r['target_outcome'] != 'ongoing'}
        if len(splits) != 1 or len(targets) > 1:
            for row, _, _ in group:
                row.update(status='excluded', reason='collapsed_record_split_or_outcome_conflict')
            continue
        shared = next(((r, e) for r, _, e in group if e is not None), None)
        for row, data, external in group:
            if external is None and shared is not None:
                origin, external = shared
                row['external_policy'] = 'shared_from_identical_canonical_record'
                row['shared_external_original_path'] = origin['original_path']
            path = output / row['record_path']
            with path.open('xb') as stream:
                stream.write(data)
            sidecar_fields = ('source_rules', 'target_rules', 'source_decisions', 'target_decisions',
                              'discarded_decisions', 'source_outcome', 'target_outcome',
                              'source_record_sha256', 'target_record_sha256', 'split_identity_sha256',
                              'original_path', 'original_file_sha256')
            receipt = {key: row[key] for key in sidecar_fields}
            save(path.with_suffix('.rules-migration.json'),
                 dict(receipt, schema='paisho-rules-migration-v1'))
            if external:
                save(path.with_suffix('.outcome.json'), external)
    save(output / 'records.json', rows)
    accepted = [row for row in rows if row['status'] == 'converted']
    summary = dict(scanned=len(rows), converted=len(accepted), excluded=len(rows)-len(accepted),
                   unique_records=len({r['target_record_sha256'] for r in accepted}),
                   shortened_records=sum(r['discarded_decisions'] > 0 for r in accepted),
                   discarded_decisions=sum(r['discarded_decisions'] for r in accepted),
                   changed_outcomes=sum(r['source_outcome'] != r['target_outcome'] for r in accepted),
                   outcomes=dict(Counter(r['target_outcome'] for r in accepted)),
                   excluded_reasons=dict(Counter(r['reason'] for r in rows if r['status'] == 'excluded')),
                   transferred_search_targets=0)
    save(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter', type=Path, default=Path('target/release/examples/revalidate_record'))
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.output, args.converter), indent=2))
