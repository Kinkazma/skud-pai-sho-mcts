#!/usr/bin/env python3
"""Plot fresh fixed-model MCTS cohorts; no collection or profile mutation."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_compact_deadlines import load_run
from optimize_compact_game_rate import analyze_run


def summarize(archive):
    plan = json.loads((archive / 'plan.json').read_text())
    curves = []
    source_versions = set()
    for budget in plan['budgets']:
        maximum = plan['maximum_seconds'][str(budget)]
        run = archive / f'retention{budget}'
        result = analyze_run(run, maximum)
        groups, _ = load_run(run)
        assert hashlib.sha256((run / 'initial-model.json').read_bytes()).hexdigest() == plan['model_sha256']
        source_versions.update(json.loads(k)['source_sha256'] for k in groups)
        observations = [r for group in groups.values() for r in group]
        assert len(observations) == plan['games_per_budget']
        assert all(not json.loads(k)['learning_enabled'] for k in groups)
        assert result['best'] is not None and not result['missing_game_ids']
        assert result['budget'] == budget and result['workers'] == plan['workers']
        fine = result['table']
        points = [r for r in fine if r['seconds'] % 5 == 0]
        previous = 0
        for point in points:
            t = point['seconds']
            # Independent reconstruction, including every interrupted attempt.
            assert not any(r.disposition == 'time-censored' and r.elapsed < t for r in observations)
            finishes = sum(r.disposition == 'terminal' and r.elapsed <= t for r in observations)
            cost = sum(min(r.elapsed, t) for r in observations)
            assert abs(point['rate'] - finishes / cost) < 1e-12
            assert finishes == point['finished']
            point['additional_finishes'] = finishes - previous
            previous = finishes
        best5 = max(points, key=lambda p: (p['rate'], -p['seconds']))
        curves.append(dict(budget=budget, maximum=maximum, attempts=len(observations),
                           best=result['best'], best_5s=best5, points=points,
                           fine_points=fine, wall_seconds=result['observed_run_wall_seconds'],
                           observed_finishes=result['observed_run_terminal_games'],
                           weights_sha256=result['model_weights_sha256'],
                           observations=[dict(file=r.identity, duration=r.elapsed,
                                              disposition=r.disposition, decisions=r.decisions)
                                         for r in observations],
                           provenance=result['provenance']))
    assert len({c['weights_sha256'] for c in curves}) == 1
    assert len(source_versions) == 1
    return curves


def fr(number, places=3):
    return f'{number:.{places}f}'.replace('.', ',')


def draw(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    colors = ['#087f8c', '#397647', '#b16a08', '#b74654', '#7654b3']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), facecolor='#fafbfd')
    fig.subplots_adjust(left=.08, right=.97, top=.86, bottom=.12, hspace=.63, wspace=.27)
    for ax, c, color in zip(axes.flat, curves, colors):
        xs = [p['seconds'] for p in c['points']]
        ys = [p['rate'] for p in c['points']]
        ax.plot(xs, ys, marker='o', markersize=4, lw=2, color=color)
        best = c['best']
        ax.scatter([best['seconds']], [best['rate']], color=color, marker='*', s=150, zorder=4)
        ax.set_title(f"MCTS-{c['budget']} · meilleur seuil observé : {fr(best['seconds'], 1)} s",
                     loc='left', fontsize=13, weight='bold', color=color, pad=12)
        ax.set_xlim(0, c['maximum'] + 2)
        ax.set_ylim(bottom=0, top=max(ys + [best['rate']]) * 1.16)
        ax.xaxis.set_major_locator(MultipleLocator(5 if c['maximum'] <= 60 else 10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fr(y)))
        ax.grid(alpha=.20)
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Coupure par partie (s)', fontsize=11)
        ax.set_ylabel('Fins / seconde de travailleur', fontsize=10)
    ax = axes.flat[-1]
    ax.axis('off')
    ax.text(0, 1, 'Les interruptions sont comptées dans le coût', weight='bold', va='top', fontsize=13)
    ax.text(0, .80, 'Débit = fins avant t ÷ somme min(durée, t)\n\n'
            '● Points tous les 5 s, sans lissage\n★ Maximum recherché par pas de 0,5 s\n\n'
            'Échelles verticales adaptées à chaque budget.\n'
            'Aucune courbe au-delà du plafond observé.', va='top', fontsize=12, linespacing=1.5)
    fig.text(.08, .95, 'Nouvelles coupures avec le moteur optimisé', fontsize=24, weight='bold', color='#172739')
    fig.text(.08, .904, '32 tentatives par budget · mêmes poids figés · 8 travailleurs · aucune mise à jour du modèle',
             fontsize=12, color='#495b70')
    fig.text(.08, .054, 'Courbes reconstruites sur les nouvelles durées ; le débit mural total de la machine est une mesure distincte.',
             fontsize=11, color='#495b70')
    fig.text(.08, .028, 'Petit échantillon et interruptions idéales : le maximum observé ne garantit pas un optimum universel à la demi-seconde.',
             fontsize=10, color='#495b70')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'debits-mcts.{suffix}', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, (total, extra) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                      gridspec_kw={'height_ratios': [1.65, 1]}, facecolor='#fafbfd')
    fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.15, hspace=.28)
    for i, (c, color) in enumerate(zip(curves, colors)):
        xs = [p['seconds'] for p in c['points']]
        ys = [p['finished'] for p in c['points']]
        extra_counts = [p['additional_finishes'] for p in c['points']]
        assert sum(extra_counts) == ys[-1]
        total.plot(xs, ys, marker='o', ms=4, lw=2, color=color, label=f"MCTS-{c['budget']}")
        extra.bar([x + (i - 2)*.75 for x in xs], extra_counts, width=.70, color=color)
    total.set_ylim(0, curves[0]['attempts'] + 2)
    total.set_ylabel('Parties terminées sur 32 tentatives')
    total.yaxis.set_major_locator(MultipleLocator(4))
    total.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.17), frameon=False, fontsize=11)
    extra.set_ylabel('Nouvelles fins par tranche de 5 s')
    extra.set_xlabel('Coupure (s) · barre à 20 s = fins supplémentaires entre 15 et 20 s')
    extra.yaxis.set_major_locator(MultipleLocator(2))
    extra.set_xlim(0, max(c['maximum'] for c in curves) + 3)
    extra.xaxis.set_major_locator(MultipleLocator(5))
    for ax in (total, extra):
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    fig.text(.08, .95, 'Combien de parties rapporte le temps supplémentaire ?', fontsize=22, weight='bold', color='#172739')
    fig.text(.08, .90, 'Nouvelles mesures · effectifs identiques · fins cumulées et gain par tranche de 5 secondes', fontsize=12, color='#495b70')
    fig.text(.08, .073, 'Les courbes s’arrêtent à leur plafond d’observation : 30 / 30 / 60 / 90 / 120 s. Une fin de courbe ne signifie pas 100 %.',
             fontsize=10, color='#495b70')
    fig.text(.08, .040, 'Pour choisir la coupure, utiliser le graphique de débit : attendre davantage augmente parfois les fins mais réduit la production par seconde.',
             fontsize=10, color='#495b70')
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'parties-et-temps-ajoute.{suffix}', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(curves, output):
    lines = ['# Coupures MCTS après génération différée', '',
             'Nouvelles parties, moteur optimisé, mêmes poids et huit travailleurs. '
             'La demande utilisateur rouvre les coupures antérieures pour cette mesure.', '',
             '![Débits](debits-mcts.png)', '', '![Quantités](parties-et-temps-ajoute.png)', '',
             '| MCTS | Seuil retenu, grille 0,5 s | Maximum au pas de 5 s | Fins au seuil / 32 | Fins / seconde de travailleur | Durée réelle du lot |',
             '|---:|---:|---:|---:|---:|---:|']
    for c in curves:
        b = c['best']
        lines.append(f"|{c['budget']}|{fr(b['seconds'],1)} s|{c['best_5s']['seconds']:g} s|{b['finished']}/32|{fr(b['rate'],5)}|{fr(c['wall_seconds'],1)} s|")
    lines += ['', '## Méthode et limites', '',
              'Pour chaque t : nombre de fins observées avant t / somme min(durée,t) '
              'sur toutes les tentatives. Les interruptions consomment du temps ; '
              'elles ne deviennent ni fins ni nulles. Aucun quota 95 % ou 75 %.', '',
              'Les courbes réutilisent les mêmes traces à chaque seuil : ce ne sont pas '
              'des lots indépendants par point. Graines communes entre budgets, mais '
              'trajectoires différentes. Moitié auto-jeu figé, moitié ancienne heuristique '
              'au même budget, départs standards, limite 2048 décisions. Les seuils ne '
              'sont pas automatiquement transférables à des départs avancés ou à un autre modèle.', '',
              'La simulation de coupure est idéale ; le moteur utilise un délai souple '
              'qui peut terminer la recherche engagée. Les résultats proches et les '
              'maxima de ce petit échantillon n’ont pas une précision universelle de 0,5 s. '
              'Le débit occupé n’est pas le débit mural mesuré de toute la machine. '
              'La différence avec les anciens lots mélange effet du moteur, nouvelles '
              'trajectoires et conditions de mesure : ce n’est pas un test causal d’accélération.', '']
    for c in curves:
        lines += [f"## MCTS-{c['budget']} : numérateurs et dénominateurs", '',
                  '| Coupure | Fins | Nouvelles fins depuis le point précédent | Temps cumulé de travailleur | Fins/s |',
                  '|---:|---:|---:|---:|---:|']
        for p in c['points']:
            lines.append(f"|{p['seconds']:g} s|{p['finished']}|{p['additional_finishes']}|{fr(p['cost'])} s|{fr(p['rate'],5)}|")
        lines.append('')
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new analysis directory')
    curves = summarize(args.archive)
    args.output.mkdir(parents=True)
    result = dict(schema='compact-new-cutoff-report-v1', curves=curves,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'points-et-traces.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    draw(curves, args.output)
    report(curves, args.output)
    print(json.dumps([dict(budget=c['budget'], seconds=c['best']['seconds'],
                           finishes=c['best']['finished'], rate=c['best']['rate']) for c in curves]))


if __name__ == '__main__':
    main()
