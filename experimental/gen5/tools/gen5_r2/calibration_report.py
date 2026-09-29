"""Describe completed R2 duration calibration; no checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyze import group_scores


def describe(point):
    result = {k: point[k] for k in ['phase', 'optimizer_steps', 'steps_in_phase']}
    for split in ['old', 'new', 'validation']:
        reads = point[split]['reads']
        groups = group_scores(reads)
        result[split] = {'wins': sum(r['win'] for r in reads), 'roots': len(reads),
                         'groups': len(groups), 'group_win_rate': float(np.mean(list(groups.values()))),
                         'mean_loss': point[split]['mean_loss']}
    return result


def analyze(result):
    assert len(result['trials']) == 6
    plan = result['plan']
    sets = [set(plan[k]) for k in ['a_groups', 'b_groups', 'validation_groups']]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert not plan['original_holdout_used']
    rows = []
    for trial in result['trials']:
        curve = trial['curve']
        assert [p['optimizer_steps'] for p in curve] == [0, 2048, 8192, 16384, 16384, 18432, 24576, 32768]
        initial, after_a, final = curve[0], curve[3], curve[-1]
        for split in ['old', 'new', 'validation']:
            assert curve[3][split] == curve[4][split]
            assert {r['group'] for r in final[split]['reads']} == sets[['old', 'new', 'validation'].index(split)]
            assert len({r['key'] for r in final[split]['reads']}) == plan['counts'][{'old':'a','new':'b','validation':'validation'}[split]]
        maps = [{r['key']: r['win'] for r in p['old']['reads']} for p in [initial, after_a, final]]
        assert maps[0].keys() == maps[1].keys() == maps[2].keys()
        acquired = [k for k in maps[0] if not maps[0][k] and maps[1][k]]
        rows.append({'seed': trial['seed'], 'mode': trial['mode'], 'seconds': trial['seconds'],
                     'curve': [describe(p) for p in curve], 'new_old_acquisitions_A': len(acquired),
                     'acquisitions_A_missing_after_B': sum(not maps[2][k] for k in acquired),
                     'all_old_successes_A_missing_after_B': sum(maps[1][k] and not maps[2][k] for k in maps[1])})
    assert {(t['mode'], t['seed']) for t in rows} == {(m,s) for m in plan['arms'] for s in plan['seeds']}
    return {'scope': 'original training groups only; inner validation, not independent confirmation',
            'original_holdout_used': False, 'counts': plan['counts'], 'trials': rows,
            'seconds': result['seconds'], 'checkpoint_selected': False,
            'publication_or_joint_actor_training_tested': False}


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    colors = {'dense': '#64748b', 'messages': '#087f8c'}
    labels = {'dense': 'Résidu dense', 'messages': 'Pièces + messages'}
    for ax, split, title in zip(axes, ['old', 'new', 'validation'],
                               ['Ancien lot · 69 positions / 49 groupes',
                                'Nouveau lot · 75 positions / 49 groupes',
                                'Validation · 32 positions / 24 groupes']):
        for mode in ['dense', 'messages']:
            trials = [t for t in report['trials'] if t['mode'] == mode]
            x = np.array([p['optimizer_steps'] for p in trials[0]['curve']])
            y = np.array([[p[split]['group_win_rate']*100 for p in t['curve']] for t in trials])
            ax.plot(x, y.mean(axis=0), color=colors[mode], label=labels[mode], marker='o', markersize=3)
            ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=colors[mode], alpha=.13)
        ax.axvline(16384, color='#475569', ls=':', lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Mises à jour du réseau expérimental')
        ax.set_xticks([0,8192,16384,24576,32768], ['0','8 192','16 384','24 576','32 768'], rotation=30)
        ax.set_ylim(0, 104); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Meilleur coup gagnant · moyenne des groupes (%)')
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Apprendre les exemples ne suffit pas à généraliser — calibration R2', fontsize=13)
    fig.text(.5, .02, 'Moyenne de 3 graines ; ruban = min–max, pas intervalle de confiance.\n'
             'Après 16 384 mises à jour : nouveau lot + 50 % de rappel. Acteur principal figé ; aucun test de publication.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.94))
    fig.savefig(out.with_suffix('.png'), dpi=160)
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path)
    args = parser.parse_args()
    assert json.loads((args.calibration/'completion.json').read_text())['complete']
    report = analyze(json.loads((args.calibration/'results.json').read_text()))
    (args.calibration/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(report, args.calibration/'learning-curves')
    for t in report['trials']:
        print(json.dumps({'seed':t['seed'],'mode':t['mode'], 'final':t['curve'][-1],
                          'acquired_A':t['new_old_acquisitions_A'],
                          'missing_after_B':t['acquisitions_A_missing_after_B']}))


if __name__ == '__main__':
    main()
