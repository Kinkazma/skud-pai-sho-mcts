#!/usr/bin/env python3
"""Convert an immutable public Skud archive through the native legal replay engine.

No search, network or source writes. Canonical PSRs are deduplicated before fit.
Declared resignations remain ongoing PSRs with an explicit external target sidecar.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def declared_side(game):
    winner = game.get('winner')
    if winner is None:
        return None
    seats = [side for side, key in [('H', 'host'), ('G', 'guest')] if winner == game[key]]
    if len(seats) != 1:
        raise ValueError('ambiguous or unknown declared winner')
    for entry in game['tournaments']:
        other = entry.get('gameWinnerUsername')
        if other and other != winner:
            raise ValueError('conflicting tournament winner')
    return seats[0]


def convert_one(game, source, output, converter):
    identity = game['game_id']
    raw = source / game['notation_path']
    row = dict(game_id=identity, original_path=str(raw),
               original_sha256=sha(raw.read_bytes()), metadata=game,
               status='excluded')
    if row['original_sha256'] != game['notation_sha256']:
        raise ValueError(f'source hash mismatch: {raw}')
    if game['options']:
        return row | dict(reason='unsupported_options')
    if game['notation_status'] != 'present_unvalidated':
        return row | dict(reason=game['notation_status'])
    psr = output / 'replayed' / f'{identity}.psr'
    result = subprocess.run([str(converter), str(raw), str(psr)], text=True, capture_output=True)
    if result.returncode:
        return row | dict(reason='conversion_or_rules_rejection', detail=result.stderr.strip())
    facts = json.loads(result.stdout)
    row.update(facts=facts, psr_path=str(psr), psr_sha256=sha(psr.read_bytes()))
    try:
        side = declared_side(game)
    except ValueError as error:
        return row | dict(reason='metadata_conflict', detail=str(error))
    if facts['decisions'] == 0:
        return row | dict(reason='no_played_decisions')
    if facts['outcome'] != 'ongoing':
        if side and facts['outcome'] != side:
            return row | dict(reason='engine_declared_outcome_conflict')
        return row | dict(status='accepted', target=facts['outcome'], target_source='rules_terminal')
    if game['result_id'] == 9 and side:
        return row | dict(status='accepted', target=side, target_source='site_resignation')
    return row | dict(reason='no_verified_terminal_result')


def prepare(source, output, converter, workers=4):
    started = time.monotonic()
    source, output, converter = source.resolve(), output.resolve(), converter.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'replayed').mkdir()
    (output / 'accepted').mkdir()
    games_raw = (source / 'games.json').read_bytes()
    games = json.loads(games_raw)
    save(output / 'plan.json', dict(schema='skud-human-conversion-v1', source=str(source),
         metadata_sha256=sha(games_raw), converter=str(converter), converter_sha256=sha(converter.read_bytes()),
         workers=workers, external_result_policy='only identity-checked site resignation result_id=9',
         result_id_evidence='sources/SkudPaiSho.af1c5e6f.js: resultId 9 displays Opponent has resigned'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda game: convert_one(game, source, output, converter), games))
    groups = {}
    for row in rows:
        if 'psr_sha256' in row:
            groups.setdefault(row['psr_sha256'], []).append(row)
    for identity, group in groups.items():
        accepted = [r for r in group if r['status'] == 'accepted']
        if not accepted:
            continue
        # A canonical sequence cannot receive different target outcomes.
        if len({r['target'] for r in accepted}) != 1 or any(r.get('reason', '').endswith('conflict') for r in group):
            for row in accepted:
                row.update(status='excluded', reason='canonical_target_conflict')
            continue
        target = output / 'accepted' / f'{identity}.psr'
        shutil.copyfile(accepted[0]['psr_path'], target)
        if accepted[0]['target_source'] == 'site_resignation':
            save(target.with_suffix('.outcome.json'), dict(schema='paisho-external-outcome-v1',
                 record_sha256=identity, outcome=accepted[0]['target'], kind='site_resignation',
                 source_records=[dict(game_id=r['game_id'], original_sha256=r['original_sha256'],
                                     metadata_sha256=sha(json.dumps(r['metadata'], sort_keys=True).encode())) for r in accepted]))
    save(output / 'records.json', rows)
    summary = dict(source_games=len(rows), accepted_source_games=sum(r['status']=='accepted' for r in rows),
                   accepted_unique_games=len(list((output/'accepted').glob('*.psr'))),
                   accepted_by_result=dict(Counter(r['target_source'] for r in rows if r['status']=='accepted')),
                   excluded_by_reason=dict(Counter(r['reason'] for r in rows if r['status']=='excluded')),
                   legal_replayed_games=sum('facts' in r for r in rows), conversion_seconds=time.monotonic()-started)
    save(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    print(json.dumps(prepare(args.source, args.output, args.converter, args.workers), indent=2))
