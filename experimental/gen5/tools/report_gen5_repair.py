#!/usr/bin/env python3
"""Recompute the integrated Gen5 V22 trial report without running training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    initial = read(root / 'recovery/resume-progress.json')
    final = read(root / 'trial/progress.json')
    rows = [json.loads(line) for line in (root / 'trial.log').read_text().splitlines()
            if line.startswith('{')]
    rows = [row for row in rows if 'lane' in row]
    keys = ('eligible_examples', 'fresh_used', 'replay_used', 'human_used',
            'durable_draws', 'durable_scheduled', 'learned_batches', 'tactical_evaluations')
    lanes = {}
    for lane in sorted({row['lane'] for row in rows}):
        selected = [row for row in rows if row['lane'] == lane]
        lanes[lane] = {'receipts': len(selected), **{key: sum(row[key] for row in selected) for key in keys},
                       'outcomes': dict(Counter(row['outcome'] for row in selected))}
    consumed = sum(row['fresh_used'] + row['replay_used'] + row['human_used'] for row in rows)
    recall = sum(row['durable_draws'] for row in rows)
    assert consumed == final['recall_quotas']['consumed_examples']
    assert recall == final['recall_quotas']['consumed_recall']
    assert recall * 2 == consumed
    assert sum(row['learned_batches'] for row in rows) == final['updates'] - initial['updates']
    assert final['replay_positions'] == initial['replay_positions'] == 327680
    assert not final['errors'] and not final['async_error']
    panels = {}
    for name in ('panel-initial', 'panel-final'):
        positions = read(root / name / 'positions.json')
        tactics = read(root / name / 'tactics.json')
        panels[name] = {'positions': len(positions), 'sources': len({p['source'] for p in positions}),
                        'policy_immediate_wins': sum(p['policy_wins'] for p in positions),
                        'tactical_positions': sum(bool(p['wins']) for p in positions),
                        'mean_policy_loss': sum(p['loss_policy'] for p in positions) / len(positions),
                        'mean_value_loss': sum(p['loss_value'] for p in positions) / len(positions),
                        'noisy_search_wins': sum(p['selected_wins'] for p in tactics),
                        'noisy_searches': len(tactics)}
    replays = [json.loads(line) for line in (root / 'core-replay.jsonl').read_text().splitlines()]
    for replay in replays:
        path = Path(replay['path'])
        receipt = read(path.with_suffix('.json'))
        assert digest(path) == receipt['psr_sha256']
        if 'outcome' in receipt:
            assert replay['outcome'] == receipt['outcome']
        else:
            seat = {'H': 'Host', 'G': 'Guest'}[receipt['candidate_seat']]
            score = (None if replay['outcome'] == 'Ongoing' else .5 if replay['outcome'] == 'Draw'
                     else float(replay['outcome'] == f'Win({seat})'))
            assert score == receipt['score']
    comparison = read(root / 'frozen-comparison/report.json')
    tests = {}
    for name in ('ai-full-tests.log', 'train-full-tests.log'):
        lines = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', (root / name).read_text())
        assert lines and all(row[0] == 'ok' and row[2] == '0' for row in lines)
        tests[name] = {'passed': sum(int(row[1]) for row in lines), 'ignored': sum(int(row[3]) for row in lines)}
    python = (root / 'dashboard-tests.log').read_text()
    assert python.endswith('OK\n')
    tests['dashboard_python'] = int(re.search(r'Ran (\d+) tests', python)[1])
    assert 'test result: ok. 1 passed;' in (root / 'bank-tests.log').read_text()
    tests['bank_coverage'] = 1
    original = Path('training-runs/micro-gen5-ui-20260910-185838-61e652/training/model.json')
    prepared = read(root / 'prepared-model.json')
    source = read(original)
    assert source['parameters'] == prepared['parameters'][:6286]
    assert prepared['updates'] == source['updates'] == initial['updates']
    assert read(root / 'recovery/replay.index.json')['rows'] == read(initial['checkpoint_replay_index'])['rows']
    result = {'single_training_trial': True, 'receipts': len(rows), 'lanes': lanes,
              'learned_examples': consumed, 'recall_examples': recall, 'recall_fraction': recall / consumed,
              'learner_steps': final['updates'] - initial['updates'], 'final_version': final['version'],
              'native_elapsed_seconds': final['elapsed_seconds'] - initial['elapsed_seconds'],
              'replay_bytes': final['replay_bytes'], 'recall_cache': final['durable_recall'],
              'recall_seconds': final['consolidation_seconds'], 'panels': panels,
              'comparison': {k: comparison[k] for k in ('wins', 'draws', 'losses', 'unknown', 'complete_pairs', 'elapsed_seconds')},
              'comparison_unknown_cycles': [x for x in replays if '/frozen-comparison/' in x['path'] and x['outcome'] == 'Ongoing'],
              'independent_psr_replays': len(replays), 'target_verification': read(root / 'trial-verification.json'),
              'tests': tests, 'original_model_sha256': digest(original), 'prepared_model_sha256': digest(root / 'prepared-model.json'),
              'trial_model_sha256': digest(root / 'trial/model.json'), 'staged': read(root / 'staged.json')}
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('receipts', 'learned_examples', 'recall_fraction', 'learner_steps', 'comparison', 'independent_psr_replays')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root.resolve())
