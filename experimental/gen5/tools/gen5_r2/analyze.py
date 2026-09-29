"""Report paired group-weighted R2 results and the prespecified intervals."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def group_scores(rows, key='win'):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['group']].append(float(row[key]))
    return {g: float(np.mean(v)) for g, v in grouped.items()}


def interval(differences):
    rng = np.random.default_rng(260926)
    d = np.asarray(differences)
    samples = d[rng.integers(len(d), size=(20000, len(d)))].mean(axis=1)
    return np.quantile(samples, [1/120, 119/120]).tolist()


def motif(rows):
    result = []
    for slot, name in enumerate(['own_off_centre', 'opponent_off_centre', 'own_touching', 'opponent_touching']):
        tp = fp = tn = fn = 0
        by_group = collections.defaultdict(lambda: [[], []])
        for r in rows:
            target = r['labels'][slot]
            if target is None:
                continue
            predicted = r['probabilities'][slot] >= .5
            tp += int(target and predicted); fn += int(target and not predicted)
            fp += int(not target and predicted); tn += int(not target and not predicted)
            by_group[r['group']][target].append(float(predicted == bool(target)))
        rates = [float(np.mean([np.mean(v[i]) for v in by_group.values() if v[i]])) for i in [0, 1]]
        result.append({'target': name, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                       'precision': None if tp+fp == 0 else tp/(tp+fp),
                       'recall': None if tp+fn == 0 else tp/(tp+fn),
                       'groups': len(by_group), 'group_weighted_specificity': rates[0],
                       'group_weighted_sensitivity': rates[1], 'group_balanced_accuracy': float(np.mean(rates))})
    return result


def analyze(path):
    result = json.loads(Path(path).read_text())
    policies = [t for t in result['trials'] if t['task'] == 'policy']
    rows, grouped = [], collections.defaultdict(list)
    for t in policies:
        old0, olda, oldb = [dict((r['key'],r['win']) for r in phase['old']) for phase in [t['initial'],t['A'],t['B']]]
        initial = group_scores(t['initial']['held']); final = group_scores(t['B']['held'])
        assert initial.keys() == final.keys()
        grouped[t['mode']].append(final)
        rows.append({'seed':t['seed'],'mode':t['mode'],'held_wins_initial':sum(r['win'] for r in t['initial']['held']),
                     'held_wins_final':sum(r['win'] for r in t['B']['held']), 'held_roots':len(t['B']['held']),
                     'held_groups':len(final),'group_win_initial':float(np.mean(list(initial.values()))),
                     'group_win_final':float(np.mean(list(final.values()))),
                     'group_mass_final':float(np.mean(list(group_scores(t['B']['held'],'winning_mass').values()))),
                     'old_successes_A_lost_B':sum(olda[k] and not oldb[k] for k in olda),
                     'new_acquisitions_A':sum(not old0[k] and olda[k] for k in olda),
                     'new_acquisitions_A_lost_B':sum(not old0[k] and olda[k] and not oldb[k] for k in olda)})
    means = {mode:{g:float(np.mean([seed[g] for seed in seeds])) for g in seeds[0]} for mode,seeds in grouped.items()}
    comparisons = []
    for other in ['dense','edges','pieces']:
        groups = sorted(means['messages'])
        assert set(groups) == set(means[other])
        diff = [means['messages'][g]-means[other][g] for g in groups]
        comparisons.append({'contrast':'messages-'+other,'groups':len(groups),'difference':float(np.mean(diff)),
                            'bonferroni_bootstrap_interval':interval(diff)})
    return {'policy':rows,'paired_primary_comparisons':comparisons,
            'motifs':[{'seed':t['seed'],'mode':t['mode'],'targets':motif(t['B']['held'])} for t in result['trials'] if t['task']=='motif'],
            'learning_seconds':result['total_seconds'],'publication_tested':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results');p.add_argument('out')
    a=p.parse_args();report=analyze(a.results)
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['paired_primary_comparisons'],indent=2))
