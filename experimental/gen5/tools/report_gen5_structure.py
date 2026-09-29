#!/usr/bin/env python3
"""Summarize frozen structural probes; no training or campaign writes."""
import argparse
import json
from pathlib import Path
import statistics as st


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    panel = [json.loads(s) for s in (root/'native/panel.jsonl').read_text().splitlines()]
    retention = json.loads((root/'native/retention.json').read_text())
    certified = json.loads((root/'certified/certified-fit.json').read_text())
    training = json.loads((root/'training-structure.json').read_text())
    tactical = [r for r in panel if r['wins']]
    searches = {}
    for budget in [64,256,512]:
        searches[budget] = sum(next(s for s in r['searches'] if s['budget']==budget)['selected'] in r['wins'] for r in tactical)
    totals = {k:sum(v[k] for v in training['lanes'].values()) for k in ['fresh','replay','human','optimizer_steps']}
    reanalysis = training['lanes']['Reanalysis']
    summaries = []
    for fraction in [0.,.1,.5]:
        rows = [r for r in retention if r['recall_fraction']==fraction]
        summary = {'recall':fraction}
        for stage in ['initial','after_a','after_b']:
            summary[stage] = {g:{k:st.mean(r[stage][g][k] for r in rows) for k in ['policy_ce','half_mse','tactical_hits','tactical_mass']} for g in ['learn','interference','heldout']}
        summaries.append(summary)
    reader = json.loads((root/'reader/reader-ablation.json').read_text())
    # Recompute the reference using the final executable and compare it to the
    # first panel before accepting the ablation's changes as reader effects.
    for r in reader:
        expected = next(s for s in panel[r['index']]['searches'] if s['budget']==512)
        assert r['with_reader']['selected']==expected['selected']
        assert r['with_reader']['simulations']==expected['simulations']
    summary = dict(
        positions=len(panel),tactical_positions=len(tactical),search_hits=searches,
        policy_tactical_hits=sum(r['policy_pick'] in r['wins'] for r in tactical),
        noisy_tactical_hits=sum(s['selected_wins'] for r in tactical for s in r['training_searches']),
        noisy_searches=sum(len(r['training_searches']) for r in tactical),
        hidden_saturated_fraction=sum(abs(h)>.99 for r in panel for h in r['hidden'])/(32*len(panel)),
        value_saturated_positions=sum(abs(r['value'])>.99 for r in panel),
        conflicting_value_policy_trunk_gradients=sum(r['value_policy_trunk_cosine']<0 for r in panel),
        median_policy_value_trunk_norm_ratio=st.median(r['policy_trunk_norm']/max(r['value_trunk_norm'],1e-15) for r in panel),
        memory_mean_l1=st.mean(r['memory_l1'] for r in panel),memory_max_l1=max(r['memory_l1'] for r in panel),
        memory_policy_changed=sum(r['policy_pick']!=r['no_reader_pick'] for r in panel),
        memory_search_changed=sum(r['with_reader']['selected']!=r['zero_reader']['selected'] for r in reader),
        memory_tactics_with=sum(r['with_reader']['wins'] for r in reader if panel[r['index']]['wins']),
        memory_tactics_without=sum(r['zero_reader']['wins'] for r in reader if panel[r['index']]['wins']),
        bonus_positions=sum(r['phase']=='HarmonyBonus' for r in panel),
        bonus_with_neighbors=sum(r['phase']=='HarmonyBonus' and r['memory_neighbors']>0 for r in panel),
        reanalysis_sample_share=sum(reanalysis[k] for k in ['fresh','replay','human'])/sum(totals[k] for k in ['fresh','replay','human']),
        reanalysis_step_share=reanalysis['optimizer_steps']/totals['optimizer_steps'],
        totals=totals,retention=summaries,
    )
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs = plt.subplots(1,3,figsize=(15.5,4.8))
    ax=axs[0]
    labels=['Départ','Après\napprentissage','Puis autres\npositions','Avec rappel\n10 %','Avec rappel\n50 %']
    runs=certified['runs']
    values=[runs[0]['initial']['winning_mass'],st.mean(r['fitted']['winning_mass'] for r in runs)]
    values += [st.mean(r['after_background']['winning_mass'] for r in runs if r['recall']==f) for f in [0,.1,.5]]
    ax.bar(range(5),[100*v for v in values],color=['#738396','#247b87','#c7744b','#4f95a1','#22686e'])
    for i,v in enumerate(values):ax.text(i,100*v+2,f'{100*v:.1f} %',ha='center',fontsize=9)
    ax.set_xticks(range(5),labels,fontsize=8);ax.set_ylim(0,100)
    ax.set_ylabel('Probabilité moyenne des coups gagnants (%)')
    ax.set_title('Acquisition et oubli de 20 tactiques',loc='left',weight='bold',fontsize=11)
    ax=axs[1]
    for f,color in [(0,'#c7744b'),(.1,'#4f95a1'),(.5,'#22686e')]:
        row=next(x for x in summaries if x['recall']==f)
        vals=[row[s]['learn']['policy_ce'] for s in ['initial','after_a','after_b']]
        ax.plot(range(3),vals,'o-',color=color,label=f'Rappel {f*100:.0f} %')
    ax.set_xticks(range(3),['Départ','Après lot A','Après lot B'])
    ax.set_ylabel('Erreur de politique sur A (plus bas = mieux)')
    ax.set_title('Rétention de 62 cibles historiques',loc='left',weight='bold',fontsize=11)
    ax.legend(frameon=False,fontsize=9)
    ax=axs[2]
    shares=[summary['reanalysis_sample_share'],summary['reanalysis_step_share']]
    ax.bar([0,1],[v*100 for v in shares],color=['#738396','#c7744b'],width=.6)
    for i,v in enumerate(shares):ax.text(i,v*100+2,f'{v*100:.1f} %',ha='center')
    ax.set_xticks([0,1],['Exemples consommés','Mises à jour'],fontsize=9);ax.set_ylim(0,70)
    ax.set_ylabel('Part des lots issus de réanalyses (%)')
    ax.set_title('Pondération réelle dans la dernière session',loc='left',weight='bold',fontsize=11)
    fig.suptitle('Gen5 finale : trois mécanismes mesurés',x=.06,ha='left',weight='bold',fontsize=16)
    fig.text(.06,.015,'Gauche et centre : essais isolés, moyenne de 3 graines, pas une mesure Elo. Droite : 4 563 reçus de production. Modèle et contrôles inchangés.',fontsize=9,color='#4b5563')
    fig.tight_layout(rect=[.02,.08,1,.92],w_pad=2.8)
    fig.savefig(root/'diagnostic.png',dpi=160)
    fig.savefig(root/'diagnostic.svg')
    print(json.dumps({k:v for k,v in summary.items() if k!='retention'},indent=2))


if __name__ == '__main__':
    main()
