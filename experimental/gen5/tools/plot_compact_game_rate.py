#!/usr/bin/env python3
"""Audit and plot archived compact-MCTS throughput every five seconds.

No game generation, model update or cutoff change. Uses the frozen cohorts
under the supplied archive and writes a new, separate evidence directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_compact_deadlines import load_run
from optimize_compact_game_rate import analyze_run


def curve(archive, budget, maximum):
    run = archive / f'retention{budget}'
    analysis = analyze_run(run, maximum, allow_incomplete=budget == 512)
    groups, _ = load_run(run)
    rows = [row for group in groups.values() for row in group]
    missing = len(analysis['missing_game_ids'])
    points = []
    previous = None
    for original in analysis['table']:
        if original['seconds'] % 5:
            continue
        t = original['seconds']
        # Recalculate explicitly from raw durations, independently of the
        # optimizer's estimator, so the figure has auditable denominators.
        finished = sum(r.disposition == 'terminal' and r.elapsed <= t for r in rows)
        known_cost = sum(min(r.elapsed, t) for r in rows)
        if any(r.disposition == 'time-censored' and r.elapsed < t for r in rows):
            raise ValueError('plot range extends beyond known elapsed censoring')
        upper_cost = known_cost + missing * t
        lower_rate = finished / upper_cost
        upper_rate = (finished + missing) / known_cost
        assert abs(lower_rate - original['selection_rate']) < 1e-12
        point = dict(seconds=t, known_terminal_games=finished,
                     planned_attempts=len(rows)+missing,
                     known_occupied_seconds=known_cost,
                     total_occupied_seconds_bounds=[known_cost, upper_cost],
                     rate_bounds=[lower_rate, upper_rate],
                     rate=lower_rate if not missing else None,
                     plotted_rate=lower_rate,
                     gain_vs_previous_percent=None,
                     additional_known_finishes=None,
                     additional_upper_cost_seconds=None)
        if previous:
            if previous['plotted_rate']:
                point['gain_vs_previous_percent'] = 100 * (lower_rate / previous['plotted_rate'] - 1)
            point['additional_known_finishes'] = finished - previous['known_terminal_games']
            point['additional_upper_cost_seconds'] = upper_cost - previous['total_occupied_seconds_bounds'][1]
        points.append(point)
        previous = point
    return dict(budget=budget, points=points,
                best_on_five_second_grid=max(points, key=lambda p:(p['plotted_rate'], -p['seconds'])),
                installed_seconds={128:17.5, 256:30, 512:55}[budget],
                missing_game_ids=analysis['missing_game_ids'],
                observations=[dict(file=r.identity, elapsed_seconds=r.elapsed,
                                   disposition=r.disposition) for r in rows],
                provenance=analysis['provenance'])


def fr(value, places=5):
    return f'{value:.{places}f}'.replace('.', ',')


def draw(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'svg.fonttype':'none', 'axes.spines.top':False,
                         'axes.spines.right':False})
    fig = plt.figure(figsize=(16,10), facecolor='#fafbfd')
    ax = fig.add_axes([.08,.29,.89,.52], facecolor='white')
    colors = {128:'#087f8c', 256:'#c57416', 512:'#7654b3'}
    labels = {128:'MCTS-128 · 48 tentatives', 256:'MCTS-256 · 32 tentatives',
              512:'MCTS-512 · 24 prévues, 2 traces absentes'}
    locations = {128:(35,.074), 256:(35,.030), 512:(80,.031)}
    for c in curves:
        budget = c['budget']
        xs = [p['seconds'] for p in c['points']]
        ys = [p['plotted_rate'] for p in c['points']]
        color = colors[budget]
        ax.plot(xs, ys, color=color, marker='o', ms=4.3, lw=2.3, label=labels[budget], zorder=3)
        if c['missing_game_ids']:
            upper = [p['rate_bounds'][1] for p in c['points']]
            ax.fill_between(xs, ys, upper, color=color, alpha=.14, zorder=1)
            ax.plot(xs, upper, color=color, lw=1.1, ls=(0,(4,3)), alpha=.8)
        p = c['best_on_five_second_grid']
        ax.scatter([p['seconds']], [p['plotted_rate']], s=95, color=color,
                   edgecolors='white', linewidths=1.5, zorder=5)
        suffix = ' (borne prudente)' if budget == 512 else ''
        ax.annotate(f"{int(p['seconds'])} s · {fr(p['plotted_rate'])} p/s{suffix}",
                    xy=(p['seconds'],p['plotted_rate']), xytext=locations[budget],
                    fontsize=12, color=color, weight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.2),
                    bbox=dict(boxstyle='round,pad=.3', fc='white', ec='none', alpha=.95))
    ax.set_xlim(0,123)
    ax.set_ylim(0,.082)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(.01))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:fr(x,2)))
    ax.tick_params(axis='x', labelsize=10, length=0, pad=9)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.grid(axis='y', color='#e0e5eb', lw=.8)
    ax.grid(axis='x', color='#eef1f5', lw=.6)
    ax.set_xlabel('Coupure choisie par partie (secondes)', labelpad=14)
    ax.set_ylabel('Parties terminées / seconde de travailleur', labelpad=12)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    fig.text(.08,.945,'Quel délai produit le plus de parties terminées ?',
             fontsize=25, weight='bold', color='#172739')
    fig.text(.08,.898,'Un point tous les 5 s · temps des interruptions inclus · mêmes poids figés',
             fontsize=14, color='#495b70')
    fig.text(.08,.856,'Débit = nombre de fins avant la coupure ÷ temps cumulé de toutes les tentatives',
             fontsize=12, color='#495b70')
    fig.text(.08,.188,'Meilleurs points au pas de 5 s : 20 s / 30 s / 70 s.',
             fontsize=13, weight='bold', color='#172739')
    fig.text(.08,.151,'Réglages actuels : 17,5 s / 30 s / 55 s. La coupure 512 est choisie par l’utilisateur (V4).',
             fontsize=11, color='#495b70')
    fig.text(.08,.110,'Zone violette : bornes dues aux 2 traces absentes à 512, pas un intervalle de confiance statistique.',
             fontsize=11, color='#495b70')
    fig.text(.08,.077,'Courbes reconstruites sur les durées archivées ; aucun nouveau lot lancé à chaque seuil. Petits échantillons.',
             fontsize=11, color='#495b70')
    fig.text(.08,.035,'La courbe 128 s’arrête à 60 s : au-delà, les données sont censurées. Le débit global de la machine n’est pas mesuré ici.',
             fontsize=10, color='#607185')
    fig.savefig(output/'debits-pas-5-secondes.png', dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(output/'debits-pas-5-secondes.svg', facecolor=fig.get_facecolor(), metadata={'Date':None})
    plt.close(fig)


def draw_counts(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'svg.fonttype':'none', 'axes.spines.top':False,
                         'axes.spines.right':False})
    fig = plt.figure(figsize=(16,10.5), facecolor='#fafbfd')
    total = fig.add_axes([.075,.47,.895,.33], facecolor='white')
    extra = fig.add_axes([.075,.18,.895,.20], facecolor='white', sharex=total)
    colors = {128:'#087f8c', 256:'#c57416', 512:'#7654b3'}
    positions = {128:(35,45), 256:(44,34), 512:(84,15)}
    selected = {128:20, 256:30, 512:70}
    for index,c in enumerate(curves):
        b = c['budget']
        xs = [p['seconds'] for p in c['points']]
        ys = [p['known_terminal_games'] for p in c['points']]
        additions = [ys[0]] + [right-left for left,right in zip(ys,ys[1:])]
        # Increments telescope to the saved cumulative terminal count.
        assert all(n>=0 for n in additions) and sum(additions)==ys[-1]
        color = colors[b]
        n = c['points'][0]['planned_attempts']
        total.plot(xs,ys,color=color,marker='o',ms=4,lw=2.3,
                   label=f'MCTS-{b} · lot de {n} parties',zorder=3)
        missing = len(c['missing_game_ids'])
        if missing:
            total.fill_between(xs,ys,[y+missing for y in ys],color=color,alpha=.14)
            total.plot(xs,[y+missing for y in ys],color=color,lw=1,ls='--',alpha=.7)
        x = selected[b]
        y = ys[xs.index(x)]
        total.annotate(f'{y} fins connues à {x} s',xy=(x,y),xytext=positions[b],
                       color=color,weight='bold',fontsize=12,
                       arrowprops=dict(arrowstyle='-',color=color,lw=1.2),
                       bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.95))
        extra.bar([x+(index-1)*1.15 for x in xs],additions,width=1.05,color=color,zorder=3)
    total.set_ylim(0,50)
    total.set_ylabel('Nombre cumulé de parties terminées',labelpad=12)
    total.yaxis.set_major_locator(MultipleLocator(10))
    total.tick_params(axis='x',labelbottom=False)
    total.legend(loc='upper right',frameon=False,fontsize=11)
    extra.set_ylim(0,23)
    extra.yaxis.set_major_locator(MultipleLocator(5))
    extra.set_ylabel('Parties supplémentaires',labelpad=12)
    extra.set_title('Ce que chaque ajout de 5 secondes rapporte',loc='left',pad=15,fontsize=15,weight='bold')
    extra.set_xlabel('Durée maximale accordée par partie (s) · barre à 20 s = gain entre 15 et 20 s',labelpad=14)
    for ax in (total,extra):
        ax.set_xlim(0,123)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.tick_params(length=0,pad=8)
        ax.tick_params(axis='x',labelsize=10)
        ax.grid(axis='y',color='#dfe5ed',lw=.8)
        ax.grid(axis='x',color='#eef1f5',lw=.6)
    fig.text(.075,.94,'Combien de parties supplémentaires rapporte l’attente ?',
             fontsize=25,weight='bold',color='#172739')
    fig.text(.075,.895,'En haut : les fins cumulées. En bas : les nouvelles fins obtenues par tranche de 5 s.',
             fontsize=14,color='#495b70')
    fig.text(.075,.85,'Données déjà enregistrées · 128 : 48 tentatives · 256 : 32 tentatives · 512 : 24 prévues, 2 traces absentes',
             fontsize=11,color='#495b70')
    fig.text(.075,.082,'Les lots ont des tailles différentes : lire l’effet du délai dans chaque courbe, sans comparer directement leurs volumes.',
             fontsize=11,color='#495b70')
    fig.text(.075,.047,'Bande violette : 0 à 2 fins possibles en plus parmi les traces absentes ; les barres comptent uniquement les fins connues.',
             fontsize=11,color='#495b70')
    fig.text(.075,.013,'Courbes reconstruites, aucune nouvelle partie. À 128, les observations s’arrêtent à 60 s ; l’absence de barres après 60 s ne signifie pas zéro fin.',
             fontsize=10,color='#607185')
    fig.savefig(output/'parties-et-temps-ajoute.png',dpi=150,facecolor=fig.get_facecolor())
    fig.savefig(output/'parties-et-temps-ajoute.svg',facecolor=fig.get_facecolor(),metadata={'Date':None})
    plt.close(fig)


def write_report(curves, output, counts=False):
    title = 'Quantité de parties et temps supplémentaire' if counts else 'Débits MCTS : lecture toutes les cinq secondes'
    picture = 'parties-et-temps-ajoute.png' if counts else 'debits-pas-5-secondes.png'
    lines = [f'# {title}', '', f'![Courbes]({picture})', '',
             'Analyse des données existantes, sans recalibration ni nouvelles parties. Les paramètres installés restent inchangés.', '',
             '## Comment vérifier le calcul', '',
             'Pour chaque durée t : compter les fins arrivées avant t, puis diviser par la somme des durées plafonnées à t de **toutes** les tentatives. Une tentative interrompue consomme du temps et ne compte pas comme fin.', '',
             'Une seconde de travailleur est une seconde occupée par un acteur de collecte. Avec huit acteurs occupés en permanence, multiplier par huit donne une projection idéale ; cela ne mesure pas les attentes ni le débit mural total de la machine.', '',
             'À 512, seules 22 des 24 traces sont disponibles. Pour les deux absentes, ni durée ni résultat ne sont inventés. Avec K(t) le coût connu et N(t) les fins connues, les bornes sont N(t)/(K(t)+2t) et (N(t)+2)/K(t). Le trait violet suit la borne inférieure. Ce ne sont pas des intervalles de confiance ; la variabilité entre parties reste en plus.', '',
             'Les points sont espacés de 5 s et reliés sans lissage. Ils sont reconstruits depuis les mêmes traces : ce ne sont pas des campagnes indépendantes exécutées pour chaque point. Les délais réels sont souples ; min(durée,t) modélise une interruption idéale.', '',
             '## Ce que les courbes établissent', '',
             '- 128 : 20 s est le meilleur point de la grille de 5 s ; 17,5 s sur la grille antérieure de 0,5 s. Après 20 s, aucune fin supplémentaire observée jusqu’à 60 s.',
             '- 256 : 30 s est le sommet observé, puis une baisse lente. Entre 30 et 35 s, aucune fin supplémentaire et environ 1,4 % de débit en moins.',
             '- 512 : le maximum de la borne prudente est à 70 s sur cette grille. Passer de 60 à 70 s ajoute une fin connue et environ 2,0 % de débit ; 69 s ne donne que 0,18 % de plus que 70 s. Les données ne démontrent pas un optimum précis à la seconde.', '',
             'Le fichier `points-et-traces.json` contient tous les numérateurs, dénominateurs, durées brutes et empreintes des sources. Les calculs de chaque point ont aussi été recomposés directement depuis ces durées et comparés au résultat de l’analyseur.', '']
    for c in curves:
        lines += [f"## MCTS-{c['budget']}", '',
                  '| Coupure (s) | Fins connues | Coût connu cumulé (s) | Débit retenu (p/s) | Borne haute (p/s) | Variation du débit depuis le point précédent |',
                  '|---:|---:|---:|---:|---:|---:|']
        for p in c['points']:
            gain = p['gain_vs_previous_percent']
            change = '—' if gain is None else f'{gain:+.2f} %'.replace('.',',')
            lines.append(f"| {p['seconds']:g} | {p['known_terminal_games']} | {fr(p['known_occupied_seconds'],3)} | {fr(p['plotted_rate'])} | {fr(p['rate_bounds'][1]) if c['missing_game_ids'] else 'identique'} | {change} |")
        if c['missing_game_ids']:
            lines += ['', 'Pour le débit prudent, ajouter 2 × la coupure au coût connu ci-dessus. Exemple à 70 s : 22 / (969,8186 + 140) = 0,019823 p/s.']
        lines.append('')
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--view',choices=('rate','counts'),default='rate')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new evidence directory')
    curves = [curve(args.archive,budget,maximum) for budget,maximum in [(128,60),(256,120),(512,120)]]
    args.output.mkdir(parents=True)
    report = dict(schema='compact-game-rate-five-second-plot-v1', step_seconds=5,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  new_games=0, cutoffs_changed=False, view=args.view, curves=curves)
    (args.output/'points-et-traces.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (draw_counts if args.view=='counts' else draw)(curves,args.output)
    write_report(curves,args.output,counts=args.view=='counts')
    print(json.dumps([{'budget':c['budget'],'best_5s':c['best_on_five_second_grid']['seconds']} for c in curves]))


if __name__ == '__main__':
    main()
