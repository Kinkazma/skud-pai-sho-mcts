#!/usr/bin/env python3
"""Read-only Gen4 preflight report, complete-game parity and independent PSR replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess


def read(p):
    return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--verifier',type=Path,required=True)
    a=p.parse_args();r=a.root
    rows=[]
    for b in [8,32,64,128,256,512,1024,2048]:
        row={'budget':b}
        for gen in [3,4]:
            d=read(r/f'budget-{b}/gen{gen}.json');g=d['games'];times=[v['seconds'] for v in g if v['terminal']]
            row[f'gen{gen}']={'terminal':len(times),'attempts':len(g),
                'mean_terminal_seconds':st.mean(times) if times else None,
                'median_terminal_seconds':st.median(times) if times else None,
                'wall_seconds':d['wall_seconds'],'terminal_games_per_wall_second':len(times)/d['wall_seconds'],
                'mean_decisions_all_attempts':st.mean(v['decisions'] for v in g)}
        rows.append(row)
    pools={}
    for t in [1,8,10]:
        ds=[read(f) for f in r.glob('clean-pool-*/gen4.json') if read(f)['threads']==t]
        pools[t]=st.mean(d['wall_seconds'] for d in ds)
    for pattern in ['abba-*','clean-pool-*']:
        dirs=sorted(f for f in r.glob(pattern) if f.is_dir())
        for d in dirs[1:]:
            for psr in dirs[0].glob('gen4-*.psr'):
                assert psr.read_bytes()==(d/psr.name).read_bytes(),f'changed game: {d}/{psr.name}'
    old=st.mean(read(r/f'abba-{i}/gen4.json')['wall_seconds'] for i in [0,3])
    new=st.mean(read(r/f'abba-{i}/gen4.json')['wall_seconds'] for i in [1,2])
    verify=r/'verified';verify.mkdir(exist_ok=True)
    cache={};records=0;facts=[]
    for f in sorted(r.rglob('*.psr')):
        if verify in f.parents:continue
        raw=f.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha not in cache:
            out=verify/f'{sha}.psr'
            # Existing verification is never a substitute for executing the verifier.
            if out.exists():out.unlink()
            done=subprocess.run([str(a.verifier),str(f),str(out)],capture_output=True,text=True,check=True)
            cache[sha]=json.loads(done.stdout)
            assert out.read_bytes()==raw,f'noncanonical {f}'
        meta=f.with_suffix('.json')
        if meta.exists():
            d=read(meta)
            if 'psr_sha256' in d:
                assert d['psr_sha256']==sha
                assert d['decisions']==cache[sha]['decisions']
            if 'outcome' in d:
                assert d['outcome']=={'host':'Win(Host)','guest':'Win(Guest)','draw':'Draw','ongoing':'Ongoing'}[cache[sha]['outcome']]
            if 'score' in d and not d.get('error'):
                outcome=cache[sha]['outcome']
                expected=None if outcome=='ongoing' else (0.5 if outcome=='draw' else float((outcome=='host')==(d['candidate_seat']=='H')))
                assert d['score']==expected,(meta,d['candidate_seat'],outcome,d['score'])
            if 'targets_sha256' in d:
                targets=f.parent/d.get('targets_file',f.stem+'.targets.json');raw_targets=targets.read_bytes()
                assert hashlib.sha256(raw_targets).hexdigest()==d['targets_sha256']
                examples=json.loads(gzip.decompress(raw_targets) if raw_targets.startswith(b'\x1f\x8b') else raw_targets)
                assert all(0<ex['decision']<=d['decisions'] for ex in examples)
                if d['termination'] not in ['rules-terminal','repetition-training-loss']:
                    assert not examples,(meta,'nonterminal target fabricated')
        facts.append({'path':str(f.relative_to(r)),'sha256':sha,**cache[sha]});records+=1
    (r/'replay-verification.json').write_text(json.dumps({'records':records,'unique_records':len(cache),'facts':facts},indent=2)+'\n')
    integration=read(r/'integration/report.json')
    targets=list((r/'integration/games').glob('*.targets.json.gz'))
    compressed=sum(f.stat().st_size for f in targets)
    plain=sum(len(gzip.decompress(f.read_bytes())) for f in targets)
    summary={'whole_games':rows,'clean_pool_mean_wall_seconds_10_games':pools,
             'selection_and_shared_features_abba':{'before':old,'after':new,'gain_percent':100*(1-new/old),'exact_psr_parity':True},
             'integration':integration,'replayed_psrs':records,'unique_psrs':len(cache),
             'target_compression':{'plain_bytes':plain,'gzip_bytes':compressed,'ratio':plain/compressed}}
    (r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for gen,color in [(3,'#8b8e98'),(4,'#1a9182')]:
        xs=[i for i,row in enumerate(rows) if row[f'gen{gen}']['median_terminal_seconds'] is not None]
        ys=[rows[i][f'gen{gen}']['median_terminal_seconds'] for i in xs]
        axes[0].plot(xs,ys,'o-',label=f'Gen{gen}',color=color)
    axes[0].set(yscale='log',ylabel='Secondes, médiane des seules fins',xlabel='Budget MCTS',xticks=range(8),xticklabels=[row['budget'] for row in rows])
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].bar([str(t) for t in pools],list(pools.values()),color='#1a9182')
    axes[1].set(xlabel='Threads CPU partagés',ylabel='Secondes pour 10 parties Gen4 / MCTS256')
    fig.suptitle('Gen4 : calcul et force restent deux mesures distinctes')
    fig.text(.02,.01,'4 parties/budget/génération ; Gen4 à32/64 : 2 fins/4 ; à8 : 0/4. Gen3 à1024 : 3/4 ; à2048 : 0/4 (limite60s).',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(r/'timings.png',dpi=160);fig.savefig(r/'timings.svg');plt.close(fig)
    svg=r/'timings.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    def secs(v):return '—' if v is None else f'{v:.3f}'
    table='\n'.join(f"| {x['budget']} | {secs(x['gen3']['median_terminal_seconds'])} ({x['gen3']['terminal']}/4) | {secs(x['gen4']['median_terminal_seconds'])} ({x['gen4']['terminal']}/4) |" for x in rows)
    text=f'''# Gen4 — prévol de campagne, 9 septembre2026

## Parties complètes : première comparaison gelée

Gen3 : `compact-human-training-2026-09-08/human-model.json` ; Gen4 :
`micro-gen4-2026-09-08/resume-smoke/model.json`. Quatre départs standards identiques
par cellule, même budget, chaque génération contre elle-même ; pool10 threads.
Limites60 secondes et600 décisions. Cette première mesure précède les deux
optimisations exactes de sélection/copie de caractéristiques ; ce ne sont pas des
confrontations de force. Les modèles produisent des trajectoires différentes.

| Budget | Gen3 médiane(s), fins/tentatives | Gen4 médiane(s), fins/tentatives |
|---:|---:|---:|
{table}

Les médianes portent uniquement sur les fins. Aucune médiane terminale Gen4 à8
ni Gen3 à2048 ; leurs parties sont interrompues, respectivement à600 décisions et
60 secondes. À32/64, exclure les deux parties inachevées rendrait une affirmation
de vitesse sans cette précision trompeuse. Moyennes, longueurs et débits effectifs
de parties terminales figurent dans `summary.json`. Petit panel, pas une garantie
universelle ni une nouvelle calibration de coupures.

## Optimisations et occupation CPU

Profil natif4s d'un véritable apprentissage : en tête des piles actives,
`has_clash`4646 observations, sélection`simulate`3089, embedding1049,
`visible_pairs`887, `relocation_creates_clash`771, `retained_bytes`518.
Ce sont des observations d'échantillonnage, pas des pourcentages de temps de campagne.
Les attentes des coordinateurs ne signifient pas que les dix threads de calcul
restent inutilisés.

PUCT évalue maintenant chaque score une fois ; caractéristiques d'actions partagées
viaArc sans copie intégrale de rapport. ABBA sur quatre parties à128 :
{old:.3f}s → {new:.3f}s par lot, soit **{100*(1-new/old):.1f}% de temps en moins** ;
PSR strictement identiques. La première exploration `pool-*` a chevauché une
compilation et n'est pas retenue. La série `clean-pool-*` est exécutée ensuite,
sans compilation ni autre entraînement : mêmes dix PSR, deux répétitions inversées,
MCTS256, {pools[1]:.3f}s avec1 thread, {pools[8]:.3f}s avec8, {pools[10]:.3f}s avec10.
Le débit est donc multiplié par{pools[1]/pools[10]:.2f} entre1 et10 sur ce lot.

La vérification des conflits/génération légale reste le premier coût observé.
Pistes ultérieures : maintenance locale exacte des relations du plateau,
projection de politique différée des feuilles jamais développées, comptage mémoire
incrémental. Elles ne sont pas annoncées comme livrées ni comme des gains acquis.
Réécrire ces parties pendant une campagne invaliderait les poids/exécutables gelés.

## Essai intégré et conservation

Le premier essai prolongé a révélé une politique vide à une feuille sans action
légale, malgré un état réglementaire encore ongoing. Correction : valeur du réseau
pour la feuille, interruption inconnue à la racine, aucun résultat terminal inventé.
Le contrôle ciblé couvre ce cas et les anciennes conventions de perspective.

Essai suivant : {integration['completed']} parties, {integration['terminal']} fins,
{integration['version']} publications, {integration['elapsed_seconds']:.2f}s incluant
la vidange après120s de calcul autorisé ; erreurs={integration['errors']}.
Le suivi Gen4/Gen4 fonctionne simultanément sur le même pool, deux coordinateurs
au plus.16 configurations standard distinctes par budget, et pas16 répétitions
des six mêmes débuts déterministes. Toute erreur du suivi remonte immédiatement.

Cibles gzip sans perte : {plain/1e6:.1f}Mo → {compressed/1e6:.1f}Mo,
réduction par{plain/compressed:.1f}. Les modèles, PSR et cibles sont tous conservés.
La reprise lit également les anciens replay JSON. {records} PSR ({len(cache)} uniques)
ont été rejoués indépendamment, avec vérification des hashes et nombres de décisions.

## Dix heures et jalons

Voir `COMPACT_MCTS_V9.md`. Un processus natif continu,36 000s, dix acteurs, pool10,
64/256 dans la proportion80/20, replayRAM4096/ratio4, départs standards. Parent :
instantané Gen4 durable de l'essai précédent, pas les modèles des essais de prévol.
Toutes les publications sont sauvegardées. Les jalons +100 internes sont des
observations provisoires contre le parent Gen4 initial au même budget, avec
régularisation et incertitude ; **pas une conversion site ni une cote Davidson
canonique**. Aucun remplacement public de Gen3. Le progrès contre Gen3 et les
humains devra être évalué séparément. Les poids Gen4 ont déjà appris sur57300
positions humaines ; ils ne sont pas restés aléatoires.
'''
    (r/'README.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='whole_games'},indent=2))


if __name__=='__main__':main()
