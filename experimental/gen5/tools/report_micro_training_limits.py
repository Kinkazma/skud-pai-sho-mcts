#!/usr/bin/env python3
"""Audit an archived micro campaign; no search, training or cutoff mutation.

The log interleaves training and evaluation games with overlapping IDs. Only
training receipts (targets_file present) enter this report. Counterfactual curves
exclude campaign-end censoring and describe a changing historical population;
they do not establish a new frozen-model or new-rule optimal timeout.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read_training_rows(path):
    rows, comparisons = [], 0
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if 'targets_file' in item:
                rows.append(item)
            else:
                comparisons += 1
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate training receipt IDs')
    return rows, comparisons


def terminal(r):
    return r['termination'] == 'rules-terminal' and r['error'] is None


def curve(rows, cutoff):
    # An early campaign shutdown does not observe the counterfactual game end.
    cohort = [r for r in rows if r['termination'] != 'wall-limit' and r['error'] is None]
    cost = sum(min(r['seconds'], cutoff) for r in cohort)
    retained = [r for r in cohort if terminal(r) and r['seconds'] <= cutoff]
    positions = sum(r['decisions'] for r in retained)
    return {'cutoff_seconds': cutoff, 'cohort_attempts': len(cohort),
            'terminal_games': len(retained), 'terminal_positions': positions,
            'occupied_worker_seconds': cost,
            'terminal_games_per_worker_second': len(retained) / cost,
            'terminal_positions_per_worker_second': positions / cost}


def audit(run):
    rows, comparisons = read_training_rows(run / 'training.log')
    report = json.loads((run / 'training/report.json').read_text())
    plan = json.loads((run / 'training/plan.json').read_text())
    parent = json.loads((run / 'initial-model.json').read_text())
    if plan.get('schema') != 'paisho-micro-selfplay-v1':
        raise ValueError('fresh-sample accounting requires the historical micro selfplay schema')
    if (len(rows) != report['completed'] or
            sum(terminal(r) for r in rows) != report['terminal'] or
            sum(r['learned_batches'] for r in rows) != report['updates'] - parent['updates']):
        raise ValueError('training log disagrees with durable report')
    kinds = Counter(r['termination'] for r in rows)
    groups = {}
    for kind in kinds:
        selected = [r for r in rows if r['termination'] == kind]
        times = sorted(r['seconds'] for r in selected)
        groups[kind] = {'games': len(selected), 'mean_seconds': statistics.mean(times),
                        'median_seconds': statistics.median(times),
                        'p95_seconds': times[int(.95 * (len(times)-1))],
                        'max_seconds': max(times), 'occupied_worker_seconds': sum(times),
                        'learned_batches': sum(r['learned_batches'] for r in selected)}
    wall = [r for r in rows if r['termination'] == 'wall-limit']
    # This audit's curve excludes only the archived final drain, not unknown
    # early per-game deadlines. Refuse this interpretation for a different run.
    if any(r['id'] < report['completed'] - plan['options']['workers'] or
           r['seconds'] >= plan['options']['game_seconds'] for r in wall):
        raise ValueError('wall-limit observations are not only short final-drain games')
    grid = [n/4 for n in range(1, 41)] + [15, 20, 25, 30]
    curves = [curve(rows, cutoff) for cutoff in grid]
    eligible = sum(r['decisions'] for r in rows if terminal(r))
    # Historical code placed fresh samples first and trained batches of 64.
    # Count fresh terminal positions used at least once, not replay multiplicity.
    used = sum(min(r['decisions'], 64*r['learned_batches']) for r in rows if terminal(r))
    sources = ['training.log', 'training/report.json', 'training/plan.json', 'initial-model.json']
    return {'schema': 'micro-training-limits-audit-v1', 'source': str(run.resolve()),
            'rules': plan.get('rules', 'skud-pai-sho-2022-03-14'),
            'hashes': {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in sources},
            'options': plan['options'], 'attempts': len(rows),
            'excluded_comparison_receipts': comparisons, 'terminations': groups,
            'actual_wall_seconds': report['elapsed_seconds'],
            'actual_terminal_games_per_wall_second': report['terminal']/report['elapsed_seconds'],
            'terminal_fresh_positions_eligible': eligible, 'terminal_fresh_positions_used': used,
            'actual_terminal_fresh_positions_used_per_wall_second': used/report['elapsed_seconds'],
            'final_drain_ids_excluded_from_curves': [r['id'] for r in wall],
            'curves': curves,
            'descriptive_best_games_grid_seconds': max(curves, key=lambda x: x['terminal_games_per_worker_second'])['cutoff_seconds'],
            'descriptive_best_positions_grid_seconds': max(curves, key=lambda x: x['terminal_positions_per_worker_second'])['cutoff_seconds'],
            'cutoff_changed': False,
            'interpretation': 'Historical evolving policy under the recorded rules, existing loop/decision stops, recorded game clocks. Curves exclude final-drain censoring, learner/I/O and restart costs. Not a measured new-campaign optimum. Terminal positions assume every completed-game decision supplies one fresh value sample; policy samples and replay exposures differ.'}


def plot(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = [p for p in summary['curves'] if p['cutoff_seconds'] <= 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    metrics = [('terminal_games_per_worker_second', 'Parties terminales / seconde de travailleur'),
               ('terminal_positions_per_worker_second', 'Positions terminales / seconde de travailleur')]
    for ax, (metric, label) in zip(axes, metrics):
        ax.plot([p['cutoff_seconds'] for p in points], [p[metric] for p in points], color='#246d91')
        peak = max(points, key=lambda x: x[metric])
        ax.scatter([peak['cutoff_seconds']], [peak[metric]], color='#b13b28', zorder=3)
        ax.annotate(f"Pic sur cette grille : {peak['cutoff_seconds']:g} s",
                    (peak['cutoff_seconds'], peak[metric]), xytext=(12, -25), textcoords='offset points')
        ax.set_xlabel('Coupure hypothétique (s) — pas de 0,25 s')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    fig.suptitle('Gen4 historique V1 : deux objectifs de débit différents')
    fig.text(.5, .035, 'Archives seules • modèles évolutifs • hors apprentissage et écritures • aucun seuil Gen5 adopté', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    for ext in ('png', 'svg'):
        path = destination / f'debits-historiques.{ext}'
        fig.savefig(path, dpi=160)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    summary = audit(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    if args.plot:
        plot(summary, args.output)
    print(json.dumps({k: summary[k] for k in ('attempts', 'descriptive_best_games_grid_seconds', 'descriptive_best_positions_grid_seconds', 'terminal_fresh_positions_used')}))


if __name__ == '__main__':
    main()
