"""Fixed R2 A/B learning and recall trials; no campaign mutation or promotion."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from . import graph
from .network import Network, prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_parity(native, rows):
    data = json.loads((native / 'parity.json').read_text())
    indexed = {r['key']: r for r in rows}
    errors = []
    for case in data['cases']:
        r = indexed[case['key']]
        mode = 'pieces' if case['mode'] == 'IndependentPieces' else 'messages'
        w = np.array(data['weights'])
        g, h, cache = graph.forward(r, w, mode)
        grad = graph.backward(r, w, mode, cache, np.array(case['output_gradient']), np.array(case['node_gradients']))
        for name, actual in [('pooled', g), ('nodes', h), ('gradient', grad)]:
            error = float(np.max(np.abs(actual-np.array(case[name]))))
            if error > 1e-11:
                raise ValueError(f'Native parity failure {mode}/{name}: {error}')
            errors.append(error)
    return {'checks': len(errors), 'max_absolute_error': max(errors)}


def measure(net, rows):
    result = []
    for r in rows:
        p, _ = net.forward(r)
        d = {'key': r['key'], 'group': r['group']}
        if net.task == 'policy':
            d.update(win=bool(r['winning'][p.argmax()]), winning_mass=float(p[r['winning']].sum()))
        else:
            d.update(probabilities=p[0].tolist(), labels=[None if not m else int(t)
                     for m, t in zip(r['mask'][0], r['targets'][0])])
        result.append(d)
    return result


def group_sample(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    return [groups[k] for k in sorted(groups)]


def pick(groups, rng):
    rows = groups[int(rng.integers(len(groups)))]
    return rows[int(rng.integers(len(rows)))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('native', type=Path); p.add_argument('plan', type=Path); p.add_argument('out', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    summary = json.loads((args.native/'summary.json').read_text())
    if sha(args.plan) != summary['plan_sha256'] or sha(args.native/'rows.jsonl') != summary['rows_sha256']:
        raise ValueError('Frozen native data/plan changed')
    rows = [prepare(json.loads(line)) for line in (args.native/'rows.jsonl').read_text().splitlines()]
    parity = native_parity(args.native, rows)
    args.out.mkdir(exist_ok=False)
    (args.out/'parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    train_groups = sorted({r['group'] for r in rows if not r['held_out']})
    a_groups = set(train_groups[::2])
    results = {'schema': 'gen5-r2-trial-v1', 'steps_per_phase': plan['steps_per_phase'],
               'phase_B_recall': .5, 'trials': [], 'data_sha256': summary['rows_sha256'],
               'no_native_integration': True, 'a_groups': sorted(a_groups)}
    start = time.perf_counter()
    for task in ['policy', 'motif']:
        train = [r for r in rows if r['task'] == task and not r['held_out']]
        held = [r for r in rows if r['task'] == task and r['held_out']]
        a = [r for r in train if r['group'] in a_groups]; b = [r for r in train if r['group'] not in a_groups]
        assert a and b and held
        ag, bg = group_sample(a), group_sample(b)
        for seed in plan['seeds']:
            # Rotate order across seeds; common row schedule for all arms.
            modes = ['dense', 'edges', 'pieces', 'messages']
            shift = plan['seeds'].index(seed)
            modes = modes[shift:] + modes[:shift]
            for mode in modes:
                net = Network(seed, mode, task)
                rng = np.random.default_rng(seed + 2000)
                t = time.perf_counter()
                initial = {'old': measure(net, a), 'held': measure(net, held)}
                for _ in range(plan['steps_per_phase']):
                    net.update([pick(ag, rng), pick(ag, rng)])
                after_a = {'old': measure(net, a), 'new': measure(net, b)}
                net.save(args.out/f'{task}-{seed}-{mode}-A.npz')
                for _ in range(plan['steps_per_phase']):
                    net.update([pick(ag, rng), pick(bg, rng)])
                after_b = {'old': measure(net, a), 'new': measure(net, b), 'held': measure(net, held)}
                net.save(args.out/f'{task}-{seed}-{mode}-B.npz')
                trial = {'task': task, 'seed': seed, 'mode': mode,
                         'parameters_allocated': sum(v.size for v in net.w.values()),
                         'initial': initial, 'A': after_a, 'B': after_b,
                         'seconds_including_evaluation_and_saves': time.perf_counter()-t}
                results['trials'].append(trial)
                (args.out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
                status = {'task': task, 'seed': seed, 'mode': mode, 'seconds': trial['seconds_including_evaluation_and_saves']}
                if task == 'policy':
                    status['held_wins'] = sum(d['win'] for d in after_b['held'])
                    status['held_roots'] = len(held)
                print(json.dumps(status), flush=True)
    results['total_seconds'] = time.perf_counter()-start
    (args.out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (args.out/'completion.json').write_text(json.dumps({'complete': True, 'trials': len(results['trials']),
                                                      'seconds': results['total_seconds']}, indent=2)+'\n')


if __name__ == '__main__':
    main()
