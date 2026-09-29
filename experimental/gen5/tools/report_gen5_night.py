#!/usr/bin/env python3
"""Read-only night review: reconcile journals, frozen panels and archived games."""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np

TZ = ZoneInfo('Europe/Paris')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cluster(values, sources):
    grouped = collections.defaultdict(list)
    for v, source in zip(values, sources):
        grouped[source].append(v)
    sums = np.array([sum(v) for v in grouped.values()])
    ns = np.array([len(v) for v in grouped.values()])
    ix = np.random.default_rng(912).integers(len(ns), size=(10000, len(ns)))
    boot = sums[ix].sum(axis=1) / ns[ix].sum(axis=1)
    return dict(mean=float(np.mean(values)), clusters=len(ns),
                interval=np.quantile(boot, [.025, .975]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    out = ap.parse_args().directory
    load = lambda p: json.loads(Path(p).read_text())
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    s = load(out / 'summary.json')
    phase = s['phases'][0]
    run = Path(phase['run'])
    manifest = load(out / 'fixed-panel-manifest.json')
    native = load(out / 'fixed-panel-results.json')
    for p, expected in load(out / 'production-hashes-before.json').items():
        assert digest(p) == expected, p
    for p, expected in load(out / 'input-hashes.json').items():
        assert digest(p) == expected, p
    for entry in manifest['models'] + manifest['positions']:
        assert digest(entry['path']) == entry['sha256'], entry['path']
    for spec in manifest['positions']:
        if spec['kind'] == 'certificate':
            proof = load(spec['path'])
            assert hashlib.sha256(proof['prefix'].encode()).hexdigest() == Path(spec['path']).stem
    models = manifest['models']
    positions = native['positions']
    rows = collections.defaultdict(dict)
    for r in native['results']:
        rows[r['model']][r['position']] = r
    assert len(native['results']) == len(models) * len(positions) == 25740
    assert all(v['updates'] == m['updates'] for v, m in zip(native['identities'], models))
    certs = [i for i, p in enumerate(manifest['positions']) if p['kind'] == 'certificate']
    wins = [i for i in certs if positions[i]['certified_actions']]
    immediate = [i for i in certs if positions[i]['immediate_actions']]
    assert (len(certs), len(wins), len(immediate)) == (384, 302, 208)
    curves = []
    for mi, model in enumerate(models):
        r = rows[mi]
        curves.append(dict(model=mi, version=model['version'], time=model['time'],
            mse=float(np.mean([(r[i]['value'] - positions[i]['target'])**2 for i in certs])),
            certified_correct=sum(r[i]['certified_selected'] for i in wins),
            certified_mass=float(np.mean([r[i]['certified_mass'] for i in wins])),
            immediate_correct=sum(r[i]['immediate_selected'] for i in immediate),
            by_outcome={str(z): dict(n=len(ids), mse=float(np.mean([(r[i]['value']-z)**2 for i in ids])))
                for z in [-1, 0, 1] for ids in [[i for i in certs if positions[i]['target'] == z]]}))
    best = max(range(len(curves)), key=lambda i: curves[i]['certified_correct'])
    comparisons = {}
    for a, label in [(0, 'initial_to_final'), (best, 'best_retained_to_final')]:
        b = len(models)-1
        def delta(ids, metric):
            return cluster([metric(rows[b][i], i)-metric(rows[a][i], i) for i in ids],
                           [positions[i]['source'] for i in ids])
        def transitions(ids, field):
            return dict(retained=sum(rows[a][i][field] and rows[b][i][field] for i in ids),
                gained=sum(not rows[a][i][field] and rows[b][i][field] for i in ids),
                lost=sum(rows[a][i][field] and not rows[b][i][field] for i in ids),
                still_missed=sum(not rows[a][i][field] and not rows[b][i][field] for i in ids))
        comparisons[label] = dict(mse=delta(certs, lambda r, i: (r['value']-positions[i]['target'])**2),
            mass=delta(wins, lambda r, i: r['certified_mass']),
            top=delta(wins, lambda r, i: float(r['certified_selected'])),
            certified=transitions(wins, 'certified_selected'),
            immediate=transitions(immediate, 'immediate_selected'))
    aliases = [i for i in range(len(positions)) if i not in certs and positions[i]['immediate_actions']]
    assert len(aliases) == 4
    assert all(not rows[mi][i]['immediate_selected'] for mi in range(len(models)) for i in aliases)
    receipts = [json.loads(l) for l in (out / 'receipts-phase-0.jsonl').open()]
    continuations = [r for r in receipts if not r['reanalysis']]
    fresh = sum(r['fresh'] for r in receipts)
    assert len(receipts) == phase['receipts'] == 330245
    assert sum(r['learned'] for r in receipts) == phase['learned'] == fresh * 5
    assert sum(r['recall'] for r in receipts) == phase['recall']
    assert sum(r['proof_recall'] for r in receipts) == phase['proof_recall']
    flow = dict(fresh=fresh, fresh_by_lane=dict(collections.Counter()),
        terminals_per_second=phase['terminals']/(phase['end']-phase['start']),
        terminals_per_second_after_first=phase['terminals']/(max(r['time'] for r in receipts)-phase['first_receipt']),
        openings=sum(r['prefix'] == 0 for r in continuations),
        reference_openings=sum(r['prefix'] == 0 and r['lane'] == 'Historical' for r in continuations),
        unique_openings=len({r['case'] for r in continuations if r['prefix'] == 0}))
    for lane in ['Selfplay', 'Historical', 'Reanalysis']:
        flow['fresh_by_lane'][lane] = sum(r['fresh'] for r in receipts if r['lane'] == lane)
    # Every new game and the original historical peak: independent core replay.
    replays = [json.loads(l) for l in (out / 'replay-results.jsonl').open()]
    for r in replays:
        game = load(Path(r['path']).with_suffix('.json'))
        assert digest(r['path']) == game['psr_sha256']
        assert r['decisions'] == game['decisions']
        seat = 'Host' if game['candidate_seat'] == 'H' else 'Guest'
        score = None if r['outcome'] == 'Ongoing' else .5 if r['outcome'] == 'Draw' else float(r['outcome'] == f'Win({seat})')
        assert score == game['score'] and game['error'] is None
    sample = load(out / 'night-sample.json')
    replay_night = [json.loads(l) for l in (out / 'night-replay-results.jsonl').open()]
    assert len(sample) == len(replay_night) == 128
    for r, game in zip(replay_night, sample):
        assert r['path'] == game['path'] and digest(r['path']) == game['psr']
        assert r['decisions'] == game['prefix'] + game['decisions']
        result = 'U' if r['outcome'] == 'Ongoing' else 'D' if r['outcome'] == 'Draw' else 'W' if r['outcome'] == f"Win({game['seat']})" else 'L'
        assert result == game['result']
    arms = {}
    for name in ['random-initial', 'random-final', 'peak-initial64', 'peak-final64', 'gen31-initial', 'gen31-final']:
        v = load(out / name / 'report.json')
        assert len(v['games']) == 32
        scores = [g['score'] for g in v['games']]
        headers = [Path(out/name/f"game-{g['id']:04}.psr").read_text().split('actions\n')[0] for g in v['games']]
        assert len(set(headers)) == 16
        if arms:
            assert headers == reference_headers
        reference_headers = headers
        arm = {k: v[k] for k in ['wins', 'draws', 'losses', 'unknown', 'complete_pairs', 'elapsed_seconds']}
        known_sum = sum(x for x in scores if x is not None)
        arm['score_bounds'] = [known_sum/32, (known_sum+arm['unknown'])/32]
        arms[name] = arm
    duels = {}
    for prefix, suffix in [('peak', '64'), ('gen31', '')]:
        a = load(out/f'{prefix}-initial{suffix}'/'report.json')['games']
        b = load(out/f'{prefix}-final{suffix}'/'report.json')['games']
        ids = [i for i in range(32) if a[i]['score'] is not None and b[i]['score'] is not None]
        duels[prefix] = dict(common_terminal_games=len(ids),
            common_score_delta=cluster([b[i]['score']-a[i]['score'] for i in ids], [i//2 for i in ids]),
            win_delta=cluster([float(b[i]['score']==1)-float(a[i]['score']==1) for i in range(32)], [i//2 for i in range(32)]))
    history = []
    for p in sorted((run/'training/history').glob('*/assessment.json')):
        v = load(p)
        initial = dict(v['references'])['initial']
        history.append(dict(sweep=p.parent.name, version=v['version'], identity=v['model'],
            conditional_elo=v.get('provisional_internal_elo_vs_initial_64'),
            **{k: initial[k] for k in ['wins','draws','losses','unknown']}))
    peak = next(h for h in history if h['version'] == 265329)
    assert (out/'historical-peak-identity.txt').read_text().strip() == peak['identity']
    # Reuse the previous measured panel, rather than generating another trial.
    previous_dir = out.parent/'gen5-campaign-review-2026-09-11'
    previous_manifest = load(previous_dir/'fixed-panel-manifest.json')
    assert previous_manifest['positions'] == manifest['positions']
    previous_native = load(previous_dir/'fixed-panel-results.json')
    previous_rows = collections.defaultdict(dict)
    for r in previous_native['results']:
        previous_rows[r['model']][r['position']] = r
    previous_curves = []
    for mi, model in enumerate(previous_manifest['models']):
        r = previous_rows[mi]
        previous_curves.append(dict(version=model['version'], time=model['time'],
            mse=float(np.mean([(r[i]['value']-positions[i]['target'])**2 for i in certs])),
            top=sum(r[i]['certified_selected'] for i in wins),
            mass=float(np.mean([r[i]['certified_mass'] for i in wins]))))
    retained = load(run/'training/checkpoint-retention.json')['models']
    largest = max(zip(models[1:-2], models[2:-1]), key=lambda pair: pair[1]['time']-pair[0]['time'])
    retention = dict(ordinals=[m['ordinal'] for m in retained],
        gap_seconds=largest[1]['time']-largest[0]['time'], gap_start=largest[0]['time'], gap_end=largest[1]['time'],
        quality_entries=sum(bool(m['results']) for m in retained))
    save('night-analysis.json', dict(flow=flow, curves=curves, comparisons=comparisons,
        best_retained=curves[best], arms=arms, duels=duels, history=history, retention=retention,
        parameter_drift=native['parameter_drift'][-1], previous_curves=previous_curves))
    save('verification.json', dict(production_files_unchanged=8, panel_inputs_hashchecked=456,
        native_evaluations=len(native['results']), native_verified_certificates=len(certs),
        new_duel_replays=192, historical_peak_replays=100, night_sample_replays=128,
        random_decisions_verified=sum(load(out/f'random-{x}-verification.json')['random_decisions_verified'] for x in ['initial','final']),
        historical_peak_identity_verified=True, new_training_updates=0, all_replays_passed=True,
        final_model_sha256=digest(run/'training/model.json')))
    # Do not draw an invented curve through the missing eight-hour checkpoint span.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    dates = [dt.datetime.fromtimestamp(m['time'], TZ) for m in models]
    split = next(i for i in range(1, len(models)) if models[i]['time']-models[i-1]['time'] > 3600)
    for ax, field, title in zip(axes, ['mse','certified_mass','certified_correct'],
        ['Erreur de valeur sur 384 positions prouvées — plus bas est meilleur',
         'Probabilité moyenne des coups gagnants certifiés — 302 positions',
         'Choix gagnants certifiés en tête de politique — sur 302']):
        for start, end in [(0, split), (split, len(models))]:
            ax.plot(dates[start:end], [c[field] for c in curves[start:end]], '.-', color='#2166ac')
        ax.axvspan(dates[split-1], dates[split], color='#eeeeee')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(alpha=.18)
    axes[1].text(.5,.5,'Poids intermédiaires non conservés\n8 h 27 sans checkpoint disponible',
                 transform=axes[1].transAxes, ha='center', color='#555555')
    axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M', tz=TZ))
    axes[-1].set_xlabel('12 septembre 2026 — heure de Paris ; panel connu de la campagne')
    fig.savefig(out/'acquis-nuit.png', dpi=180)
    fig.savefig(out/'acquis-nuit.svg')
    plt.close(fig)
    hourly = collections.defaultdict(list)
    for r in continuations:
        if r['lane'] == 'Historical':
            hourly[(r['generation'], int((r['time']-phase['start'])//3600))].append(r)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for g in sorted(s['trends']):
        cells = [(hour, rs) for (gen, hour), rs in sorted(hourly.items()) if gen == g and len(rs) >= 30]
        ax.plot([h+.5 for h,_ in cells], [100*sum(r['result']=='W' for r in rs)/len(rs) for _,rs in cells], '.-', label=g)
    ax.set(xlabel='Heures depuis le début de campagne', ylabel='Victoires / toutes les tentatives (%)',
           title='Résultats d’entraînement : faibles fluctuations, aucun passage au palier suivant', ylim=(0, 18))
    ax.legend(ncol=5, loc='upper right')
    ax.grid(alpha=.18)
    fig.savefig(out/'victoires-nuit.png', dpi=180)
    fig.savefig(out/'victoires-nuit.svg')
    plt.close(fig)
    print(json.dumps(dict(initial=curves[0], final=curves[-1], duels=duels, verification='passed'), indent=2))


if __name__ == '__main__':
    main()
