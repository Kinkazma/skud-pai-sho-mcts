#!/usr/bin/env python3
"""Archive public Skud tournament games using the endpoints of the official client.

No account, enumeration of unlisted ids, training, or remote write. Cached
responses are immutable and hashed; rerun to resume interrupted downloads.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html import escape, unescape
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://skudpaisho.com/'
SCHEMA = 'public-skud-tournament-corpus-v1'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json,text/plain,*/*'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def integer(value):
    try:
        return int(value) if str(value).strip() else None
    except (ValueError, TypeError):
        return None


def parse_game_info(raw, game_id):
    text = raw.decode('utf-8-sig').strip()
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError('expected one public game-info row')
    fields = rows[0].split('|||')
    if len(fields) < 15 or integer(fields[0]) != game_id:
        raise ValueError('game-info id/columns mismatch')
    if integer(fields[1]) != 1 or fields[2] != 'Skud Pai Sho':
        raise ValueError('game-info is not Skud Pai Sho')
    # The PHP endpoint HTML-escapes JSON quotes for some rule-option lists.
    options = json.loads(unescape(fields[8])) if fields[8].strip() else []
    if not isinstance(options, list):
        raise ValueError('invalid game option list')
    return dict(game_id=game_id, game_type_id=1, game_type=fields[2],
                host=fields[3], guest=fields[5], options=options,
                winner=fields[9] or None, result_id=integer(fields[10]),
                site_timestamp=fields[11] or None,
                host_rating=integer(fields[12]), guest_rating=integer(fields[13]),
                ranked_raw=fields[14] or None,
                rating_time_semantics='unknown: public client calls these hostRating/guestRating; historical timing not certified',
                has_game_clock=bool(fields[15]) if len(fields) > 15 else False)


def notation_status(text):
    if not text:
        return 'empty'
    if not text.startswith(('0H.', '0G.')):
        return 'unrecognized_header'
    if '0H.' not in text or '0G.' not in text:
        return 'setup_incomplete'
    return 'present_unvalidated'


def parse_tournament_index(raw):
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('tournamentList'), list):
        raise ValueError('invalid public tournament index')
    entries = data['tournamentList']
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('invalid public tournament entry')
    ids = [integer(entry.get('id')) for entry in entries]
    if any(tid is None or tid <= 0 for tid in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid or duplicate tournament ids')
    return data


def write_index(output, records, summary):
    """Standalone local index; source strings are always escaped."""
    rows = []
    labels = {'empty': 'Notation vide', 'setup_incomplete': 'Préparation incomplète',
              'unrecognized_header': 'Format à examiner', 'present_unvalidated': 'À valider'}
    for r in records:
        names = ', '.join(sorted({t['tournament_name'] for t in r['tournaments']}))
        rating = lambda value: '—' if value is None else str(value)
        rows.append('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in (
            names, r['game_id'], r['site_timestamp'] or '—',
            f"{r['host']} ({rating(r['host_rating'])})",
            f"{r['guest']} ({rating(r['guest_rating'])})", r['winner'] or '—',
            labels[r['notation_status']], r['notation_entries'])) +
            f'<td><a href="{escape(r["notation_path"], quote=True)}">Notation</a> · '
            f'<a href="{escape(r["replay_url"], quote=True)}">Site</a></td></tr>')
    document = '''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parties de tournois Skud Pai Sho</title>
<style>body{font:15px system-ui;margin:2rem;color:#182c32;background:#fafaf6}
h1{font-size:1.8rem}p{max-width:85ch;line-height:1.5}input{font:inherit;padding:.7rem;width:min(90%,38rem)}
table{border-collapse:collapse;margin-top:1.2rem;width:100%;background:white}
td,th{text-align:left;padding:.6rem;border-bottom:1px solid #ddd}th{background:#e7eeeb}
a{color:#14614d}td:last-child{white-space:nowrap}.scroll{overflow:auto}</style>
<h1>Parties de tournois Skud Pai Sho</h1>
''' + f'<p>{summary["downloaded_games"]} entrées archivées, provenant de {summary["tournaments_with_skud"]} tournois. ' + '''
Les liens « Notation » ouvrent les fichiers locaux, au format original du site (pas PSR).
Les cotes entre parenthèses sont celles fournies par le site ; leur date de référence n'est pas certifiée.
La présence d'une notation ne garantit pas que la partie est complète ou conforme aux règles actuelles.</p>
<p><a href="README.md">Documentation</a> · <a href="games.json">Métadonnées complètes</a> ·
<a href="summary.json">Bilan de collecte</a></p>
<label for="search">Rechercher un tournoi, un joueur ou un numéro de partie</label><br>
<input id="search" type="search" placeholder="Nom du tournoi ou du joueur…"><p id="count"></p>
<div class="scroll"><table><thead><tr><th>Tournoi</th><th>Partie</th><th>Date fournie</th>
<th>Host (cote)</th><th>Guest (cote)</th><th>Vainqueur déclaré</th><th>Notation</th>
<th>Entrées de notation¹</th><th>Ouvrir</th></tr></thead><tbody>
''' + '\n'.join(rows) + '''</tbody></table></div>
<p>¹ Entrées séparées par un point-virgule, préparation comprise : ce nombre n'est pas un nombre de tours complets.</p>
<script>const rows=[...document.querySelectorAll('tbody tr')];
const input=document.getElementById('search'),count=document.getElementById('count');
function filter(){let n=0;const q=input.value.toLocaleLowerCase();for(const r of rows){
r.hidden=!r.textContent.toLocaleLowerCase().includes(q);if(!r.hidden)n++;}
count.textContent=n+' / '+rows.length+' entrées affichées';}
input.addEventListener('input',filter);filter();</script></html>'''
    (output / 'index.html').write_text(document)


def write_documentation(output, summary):
    statuses = summary['notation_statuses']
    text = f'''# Corpus public de tournois Skud Pai Sho

Collecte terminée le {summary['completed_utc']}.

- {summary['indexed_tournaments']} tournois dans l'index public du site, toutes variantes confondues.
- {summary['tournaments_with_skud']} tournois contiennent des parties Skud.
- {summary['downloaded_games']} / {summary['unique_listed_skud_games']} entrées Skud archivées, sans doublon d'identifiant.
- Statuts des notations : {json.dumps(statuses, ensure_ascii=False)}.
- {summary['games_with_both_ratings']} entrées comportent deux cotes fournies par le site.
- {len(summary['errors'])} erreurs de collecte ; bilan complet dans `summary.json`.

## Consulter les parties

Ouvrir [index.html](index.html) : tableau filtrable par tournoi, joueur ou numéro.
Les fichiers `notations/ID.txt` sont les notations originales du site après
suppression des espaces extérieurs et ajout d'un saut de ligne final. **Ce ne sont
pas des PSR.** Les octets exacts des réponses restent dans `raw/games/`.
`games.json` relie chaque partie à ses tournois, joueurs, résultat déclaré,
options, cotes, URL de consultation et empreintes SHA-256.

## Portée et limites

Le périmètre est le cliché de l'index public des tournois, pas toutes les parties
jamais jouées sur le site. Les tournois mixtes sont inclus mais seules leurs
parties de type exact « Skud Pai Sho » sont téléchargées. Ce type est revérifié
dans les informations de chaque partie. Les tournois en cours sont inclus.

Une entrée vide ou une préparation incomplète est une donnée source conservée,
pas une partie complète récupérée. `present_unvalidated` décrit seulement la
présence des deux préparations dans la notation ; cela ne certifie ni terminaison
ni légalité. Les options non standard sont conservées et signalées.
Les entrées de notation comprennent la préparation et les sous-décisions :
leur nombre n'est pas un nombre de tours complets.

La date de référence des cotes `hostRating` / `guestRating` n'est pas certifiée :
elles ne sont pas assimilées automatiquement à des Elo historiques au moment du
match. Le champ de classement est conservé brut. Aucune modification de notre
échelle Elo, aucun entraînement et aucune conversion/relecture par notre moteur
n'ont été réalisés. `training_eligible` reste faux jusqu'à cette validation.

## Provenance et intégrité

Source : https://skudpaisho.com/ ; appels GET publics utilisés par son client :

- `backend/getCurrentTournaments.php?u=0`
- `backend/getTournamentInfo.php?t=ID`
- `backend/getGameInfoV2.php?userId=0&gameId=ID&isWeb=1`
- `backend/getGameNotation.php?q=ID`

Aucun compte, identifiant de partie deviné, ni écriture distante. Chaque réponse
archivée possède un reçu `.http.json` avec URL, date de réception et SHA-256.
`plan.json` conserve les paramètres initiaux ; `runs/` conserve les paramètres et
le code de chaque reprise. Le code initial et le client public sont dans `sources/`
lorsqu'ils ont été archivés lors de la découverte. Les réponses rejetées par
une version du parseur et retentées sont conservées sous `rejected/`.

Depuis ce dossier, vérifier l'intégrité avec :

```sh
shasum -a 256 -c MANIFEST.sha256
```

L'outil `tools/download_skud_tournaments.py` est reprenable : les réponses déjà
vérifiées sont relues localement. Maximum seize travailleurs et huit débuts
de requête par seconde au total ; les erreurs temporaires provoquent une attente.
Pour un nouveau cliché du site, choisir un **nouveau dossier** afin de conserver
celui-ci. Il n'y a aucune tâche récurrente.
'''
    (output / 'README.md').write_text(text)


def write_manifest(output):
    paths = sorted(p for p in output.rglob('*') if p.is_file() and
                   p.name != 'MANIFEST.sha256' and not p.name.endswith('.tmp'))
    (output / 'MANIFEST.sha256').write_text(''.join(
        f'{sha(p.read_bytes())}  {p.relative_to(output).as_posix()}\n' for p in paths))


class Archive:
    def __init__(self, output, interval):
        self.output = output
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0

    def validated_fetch(self, relative, endpoint, params, validator):
        # Preserve malformed successful responses, then make one fresh attempt.
        for attempt in range(2):
            raw = self.fetch(relative, endpoint, params)
            try:
                return validator(raw)
            except (ValueError, KeyError, TypeError):
                if attempt:
                    raise
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                old = self.output / relative
                rejected = self.output / 'rejected' / stamp / relative
                rejected.parent.mkdir(parents=True, exist_ok=True)
                old.replace(rejected)
                old.with_suffix(old.suffix + '.http.json').replace(
                    rejected.with_suffix(rejected.suffix + '.http.json'))


    def fetch(self, relative, endpoint, params=None, allow_empty=False):
        url = BASE + endpoint + ('?' + urlencode(params) if params else '')
        path = self.output / relative
        receipt = path.with_suffix(path.suffix + '.http.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and receipt.exists():
            meta = json.loads(receipt.read_text())
            raw = path.read_bytes()
            if meta['url'] != url or meta['sha256'] != sha(raw):
                raise ValueError(f'cached response changed: {relative}')
            return raw
        # An interrupted write without its receipt is not considered a verified cache.
        last_error = None
        for attempt in range(3):
            with self.lock:
                delay = max(0.0, self.next_request - time.monotonic())
                if delay:
                    time.sleep(delay)
                self.next_request = time.monotonic() + self.interval
            try:
                with urlopen(Request(url, headers=HEADERS), timeout=30) as response:
                    raw = response.read()
                    content_type = response.headers.get('Content-Type', '')
                    status = response.status
                if (not raw.strip() and not allow_empty) or b'<html' in raw[:500].lower() or b'<!doctype html' in raw[:500].lower():
                    raise ValueError(f'empty or HTML response: {relative}')
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_bytes(raw)
                temp.replace(path)
                save_json(receipt, dict(url=url, fetched_utc=datetime.now(timezone.utc).isoformat(),
                                        status=status, content_type=content_type,
                                        bytes=len(raw), sha256=sha(raw)))
                return raw
            except HTTPError as error:
                last_error = error
                if error.code not in (429, 500, 502, 503, 504):
                    raise
                retry = error.headers.get('Retry-After', '')
                time.sleep(min(60, int(retry)) if retry.isdigit() else 2**(attempt + 1))
            except (URLError, TimeoutError) as error:
                last_error = error
                time.sleep(2**(attempt + 1))
        raise RuntimeError(f'request failed after retries: {url}: {last_error}')


def download(output, workers=2, interval=.5):
    if not 1 <= workers <= 16 or interval < .125:
        raise ValueError('use 1..16 workers and at least 0.125 seconds between requests')
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get('schema') != SCHEMA or plan.get('base_url') != BASE:
            raise ValueError('incompatible existing corpus')
    else:
        save_json(plan_path, dict(schema=SCHEMA, base_url=BASE,
                  created_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
                  request_interval_seconds=interval, game_type='Skud Pai Sho',
                  scope='all games explicitly listed by public tournament index/details, including mixed events',
                  authentication='none; userId=0 guest read-only view',
                  source_client='https://skudpaisho.com/SkudPaiSho.af1c5e6f.js',
                  code_sha256=sha(Path(__file__).read_bytes()), training_performed=False))
    runs = output / 'runs'
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save_json(runs / (stamp + '.json'), dict(started_utc=stamp, workers=workers,
              request_interval_seconds=interval, code_sha256=sha(Path(__file__).read_bytes())))
    (runs / (stamp + '.py')).write_bytes(Path(__file__).read_bytes())
    archive = Archive(output, interval)
    index = archive.validated_fetch('raw/tournaments.json', 'backend/getCurrentTournaments.php', {'u': 0}, parse_tournament_index)
    tournaments = index['tournamentList']
    ids = [integer(t['id']) for t in tournaments]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError('invalid or duplicate tournament ids')
    errors = []
    details = {}
    save_json(output / 'progress.json', dict(phase='tournaments', total=len(ids), completed=0))

    def tournament(tid):
        def validate(raw):
            data = json.loads(raw)
            if (not isinstance(data, dict) or data.get('id') != tid or
                    not isinstance(data.get('games'), list) or
                    any(not isinstance(game, dict) for game in data['games']) or
                    not isinstance(data.get('name'), str) or not isinstance(data.get('status'), str)):
                raise ValueError('tournament id/games mismatch')
            return data
        return archive.validated_fetch(f'raw/tournaments/{tid}.json', 'backend/getTournamentInfo.php', {'t': tid}, validate)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(tournament, tid): tid for tid in ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                details[tid] = future.result()
            except Exception as error:
                errors.append(dict(kind='tournament', id=tid, error=str(error)))
            save_json(output / 'progress.json', dict(phase='tournaments', total=len(ids), completed=len(details), errors=errors))
    games = defaultdict(list)
    summaries = []
    types = Counter()
    for tid, data in sorted(details.items()):
        selected = []
        for game in data['games']:
            types[game.get('gameType', 'unknown')] += 1
            if game.get('gameType') == 'Skud Pai Sho':
                gid = integer(game.get('gameId'))
                if gid is None or gid <= 0:
                    raise ValueError('invalid listed Skud game id')
                games[gid].append(dict(tournament_id=tid, tournament_name=data['name'], **game))
                selected.append(gid)
        summaries.append(dict(id=tid, name=data['name'], status=data['status'],
                              forum_url=data.get('forumUrl'), listed_games=len(data['games']),
                              skud_game_ids=sorted(set(selected))))
    save_json(output / 'tournaments.json', summaries)
    save_json(output / 'listed-skud-games.json', {str(k): v for k, v in sorted(games.items())})
    print(json.dumps(dict(tournaments=len(details), tournaments_with_skud=sum(bool(t['skud_game_ids']) for t in summaries),
                          unique_skud_games=len(games), other_game_types=dict(types))), flush=True)
    records = {}

    def game(gid):
        info = archive.validated_fetch(f'raw/games/{gid}.info.txt', 'backend/getGameInfoV2.php',
                            {'userId': 0, 'gameId': gid, 'isWeb': 1}, lambda raw: parse_game_info(raw, gid))
        raw = (output / f'raw/games/{gid}.info.txt').read_bytes()
        notation = archive.fetch(f'raw/games/{gid}.notation.txt', 'backend/getGameNotation.php', {'q': gid}, allow_empty=True)
        text = notation.decode('utf-8-sig').strip()
        normalized = output / 'notations' / f'{gid}.txt'
        normalized.parent.mkdir(exist_ok=True)
        normalized.write_text(text + '\n')
        info.update(tournaments=games[gid], notation_path=f'notations/{gid}.txt',
                    notation_sha256=sha((text + '\n').encode()), raw_info_sha256=sha(raw),
                    raw_notation_sha256=sha(notation),
                    notation_entries=len([x for x in text.split(';') if x]),
                    notation_status=notation_status(text),
                    replay_url=BASE + '?watchGame=' + str(gid),
                    standard_options_candidate=not info['options'],
                    engine_validation='not run; original site notation, not PSR',
                    training_eligible=False)
        return info

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(game, gid): gid for gid in sorted(games)}
        for i, future in enumerate(as_completed(futures), 1):
            gid = futures[future]
            try:
                records[gid] = future.result()
            except Exception as error:
                errors.append(dict(kind='game', id=gid, error=str(error)))
            save_json(output / 'progress.json', dict(phase='games', total=len(games),
                      processed=i, downloaded=len(records), errors=errors))
            if i % 50 == 0 or i == len(games):
                print(json.dumps(dict(processed=i, total=len(games), downloaded=len(records), errors=len(errors))), flush=True)
    ordered = [records[k] for k in sorted(records)]
    save_json(output / 'games.json', ordered)
    notation_groups = defaultdict(list)
    for record in ordered:
        notation_groups[record['notation_sha256']].append(record['game_id'])
    summary = dict(schema=SCHEMA, completed_utc=datetime.now(timezone.utc).isoformat(),
                   indexed_tournaments=len(ids), downloaded_tournaments=len(details),
                   tournaments_with_skud=sum(bool(t['skud_game_ids']) for t in summaries),
                   unique_listed_skud_games=len(games), downloaded_games=len(records),
                   games_with_both_ratings=sum(r['host_rating'] is not None and r['guest_rating'] is not None for r in ordered),
                   games_with_ranked_field=sum(bool(r['ranked_raw']) for r in ordered),
                   standard_options_candidates=sum(r['standard_options_candidate'] for r in ordered),
                   notation_statuses=dict(Counter(r['notation_status'] for r in ordered)),
                   result_ids=dict(Counter(str(r['result_id']) for r in ordered)),
                   duplicate_notation_groups=[v for v in notation_groups.values() if len(v) > 1],
                   errors=errors, complete=not errors and len(records) == len(games),
                   scope='public tournament index snapshot only; not all games ever played on the site',
                   historical_elo_certified=False, training_performed=False)
    save_json(output / 'summary.json', summary)
    write_index(output, ordered, summary)
    write_documentation(output, summary)
    if summary['complete']:
        write_manifest(output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--request-interval', type=float, default=.5)
    args = parser.parse_args()
    result = download(args.output.resolve(), args.workers, args.request_interval)
    if not result['complete']:
        raise SystemExit('Some public records could not be downloaded; see summary.json and rerun to resume.')


if __name__ == '__main__':
    main()
