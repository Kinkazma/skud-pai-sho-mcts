#!/usr/bin/env python3
"""Gen5 dashboard with read-only status and explicit authenticated local actions.

Only small producer receipts and new journal lines are read. No archive scan, training command, model
load, PSR replay or export runs on an HTTP refresh. The bounded game selection is
cached for 30 seconds; status is cached for five. All PSRs remain in the campaign.
"""
import argparse
import html
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

if __package__:
    from .paisho_gen5_seats import SeatResults, describe_game
    from .paisho_gen5_hour import HourResults
    from .paisho_gen5_timeline import EloTimeline
    from .paisho_gen5_actions import ControlSession
    from .paisho_throughput import CounterRates
else:
    from paisho_gen5_seats import SeatResults, describe_game
    from paisho_gen5_hour import HourResults
    from paisho_gen5_timeline import EloTimeline
    from paisho_gen5_actions import ControlSession
    from paisho_throughput import CounterRates


def small_json(path, limit=131072):
    try:
        with path.open('rb') as f:
            data=f.read(limit+1)
        if len(data)>limit:raise ValueError('oversized status receipt')
        return json.loads(data)
    except FileNotFoundError:
        return {}


class Gen5Reader:
    def __init__(self, campaign):
        self.campaign=Path(campaign).resolve()
        self.training=self.campaign/'training'
        self.lock=threading.RLock()
        self.cached=None;self.next_status=0.;self.next_games=0.;self.games=[]
        self.options={};self.plan={};self.highest=0
        self.seat_results=SeatResults(self.campaign/'training.log')
        self.hour_results=HourResults(self.campaign)
        self.elo_timeline=EloTimeline(self.campaign)
        self.resume_loaded=False
        self.resume={}
        self.counter_rates=CounterRates()

    def read(self):
        with self.lock:
            now=time.monotonic()
            if self.cached is not None and now<self.next_status:return self.cached
            if not self.options:self.options=small_json(self.campaign/'config.json')
            if not self.plan:self.plan=small_json(self.campaign/'plan.json')
            if not self.resume_loaded:
                if self.options.get('resume_progress'):
                    prior=small_json(Path(self.options['resume_progress']),2_000_000)
                    self.resume=prior
                    self.seat_results.processed=prior['completed']
                    # Native checkpoints need not contain controller-derived seat
                    # totals. Keep the cumulative offset, but do not invent old
                    # model-relative results: snapshot() marks them incomplete.
                    self.seat_results.lanes=prior.get('seat_results') or {}
                self.resume_loaded=True
            state=small_json(self.campaign/'status.json')
            live_progress=small_json(self.training/'progress.json',2_000_000)
            initializing=not live_progress and state.get('state') in ('starting','running')
            progress=dict(live_progress or self.resume)
            if initializing:
                deadline=state.get('end_unix_seconds',self.plan.get('end_unix_seconds'))
                if deadline is not None:progress['remaining_seconds']=max(0,deadline-time.time())
            self.highest=max(self.highest,progress.get('last_game_id',0))
            binary_hash=self.plan.get('hashes',{}).get('bin/paisho-gen5')
            self.seat_results.update(progress.get('completed',0),self.options.get('seed'),binary_hash)
            wall_now=time.time()
            hour=self.hour_results.update(self.seat_results.offset,self.options.get('seed'),binary_hash,now=wall_now)
            throughput=self.counter_rates.observe(
                progress.get('elapsed_seconds'),
                {k:v['terminal'] for k,v in progress.get('lanes',{}).items() if k!='Reanalysis' and 'terminal' in v},
                self.resume.get('elapsed_seconds',0),
                {k:v.get('terminal',0) for k,v in self.resume.get('lanes',{}).items()},
                exclude_loading=True, active_start=live_progress.get('computation_start_elapsed_seconds'))
            if now>=self.next_games:
                # Direct bounded paths, never enumerate an ever-growing directory.
                rows=[]
                for i in range(self.highest,max(-1,self.highest-64),-1):
                    row=small_json(self.training/'games'/f'game-{i:07}.json')
                    if row and not row.get('reanalysis'):rows.append({**{k:row.get(k) for k in ('id','lane','collector_version','model_version','termination','outcome','decisions','seconds','reference_budget','opponent','reference_identity','fresh_used','prefix_decisions','continuation_decisions','case')},**describe_game(row,self.options.get('seed'),binary_hash)})
                selected=rows[:8]
                for recent in self.seat_results.recent.values():
                    selected.extend({**row,**describe_game(row,self.options.get('seed'),binary_hash)} for row in recent)
                selected.extend({**row,**describe_game(row,self.options.get('seed'),binary_hash)} for row in self.seat_results.last_decisive.values())
                for outcome in ('Win(Host)','Win(Guest)','Draw','Ongoing'):
                    candidates=[r for r in rows if r['outcome']==outcome]
                    if candidates:
                        selected.extend([min(candidates,key=lambda r:r['decisions']),max(candidates,key=lambda r:r['decisions'])])
                fields=('id','lane','collector_version','model_version','termination','outcome','decisions','seconds','reference_budget','opponent','reference_identity','fresh_used','prefix_decisions','continuation_decisions','case','gen5_seat','result_label')
                self.games=sorted({r['id']:{k:r.get(k) for k in fields} for r in selected}.values(),key=lambda r:r['id'],reverse=True)
                self.next_games=now+30
            # At most 64 tiny assessment receipts, independent of PSR/model count.
            latest={};evaluation={};internal={}
            interval=self.options.get('history_interval',900)
            upper=max(int(progress.get('elapsed_seconds',0)/interval)+1 if interval>0 else 1,
                      self.resume.get('next_history_index',0)+1)
            for i in range(upper-1,max(-1,upper-65),-1):
                directory=self.training/'history'/f'sweep-{i:04}'
                assessment=small_json(directory/'assessment.json',2_000_000)
                if not internal:
                    reference=small_json(directory/'initial'/'report.json',2_000_000)
                    if reference:
                        wins,draws,losses=(reference.get(k,0) for k in ('wins','draws','losses'))
                        internal={'index':i,'version':assessment.get('version'),'report':reference,
                                  'elo':400*math.log10((wins+0.5*draws+1)/(losses+0.5*draws+1)) if wins+draws+losses else None}
                if not evaluation and (directory/'initial'/'plan.json').exists():
                    done=resolved=planned=0
                    references=[('initial',2*self.options.get('history_initial_pairs',4))]
                    if assessment.get('scope')!='internal-only':references.append(('gen3-1',4))
                    for reference,fallback in references:
                        comparison_plan=small_json(directory/reference/'plan.json')
                        count=2*comparison_plan.get('options',{}).get('pairs',fallback//2)
                        planned+=count
                        for game in range(count):
                            row=small_json(directory/reference/f'game-{game:04}.json')
                            if row:
                                done+=1
                                resolved+=int(row.get('score') is not None)
                    evaluation={'index':i,'processed':done,'resolved':resolved,'planned':planned,
                                'complete':(directory/'assessment.json').exists()}
                latest=assessment
                if latest:break
            elapsed=progress.get('elapsed_seconds',0)
            lanes=progress.get('lanes',{})
            usage=progress.get('secondary_usage') or {}
            capacity=usage.get('reserved_seconds',0)*self.options.get('secondary_threads',1)
            denominator=usage.get('elapsed_seconds',0)*self.options.get('threads',10)
            self.cached={'campaign':str(self.campaign),'initializing':initializing,'state':state,'progress':progress,'options':{k:self.options.get(k) for k in ('threads','actors','secondary_threads','historical_capacity_fraction','budgets','game_seconds','historical_seconds','mode','history_interval','case_curriculum','checkpoint_seconds','legacy_replay','archive_workers','structural_repair','recall_fraction','opponents','checkpoint_keep','proof_recall')},
                'session_elapsed_seconds':max(0,progress.get('elapsed_seconds',0)-self.resume.get('elapsed_seconds',0)),
                'end_local':(state['end_unix_seconds']*1000 if state.get('end_unix_seconds') else self.plan.get('end_local')),'games':self.games,'assessment':latest,'evaluation_progress':evaluation,'internal_evaluation':internal,
                'gen5_terminal_per_second':sum(v or 0 for k,v in throughput['terminal_per_second'].items() if k!='Historical' or self.options.get('legacy_replay')) if throughput['ready'] and not initializing else None,
                'historical_terminal_per_second':throughput['terminal_per_second'].get('Historical',0) if throughput['ready'] and not initializing else None,
                'throughput_window':throughput,
                'historical_reserved_capacity_percent':100*capacity/denominator if denominator else None,
                'progress_age_seconds':max(0,time.time()-(self.training/'progress.json').stat().st_mtime) if (self.training/'progress.json').exists() else None,
                'seat_results':self.seat_results.snapshot(lanes),
                'last_hour':hour,
                'hourly_results':self.hour_results.history(),
                'replay_ten_minutes':self.hour_results.replay_history(),
                'elo_timeline':self.elo_timeline.read(upper),
                'read_at_unix_seconds':time.time()}
            self.next_status=now+5
            return self.cached

    def warm_history(self):
        """One bounded startup slice; no training writes or HTTP polling."""
        with self.lock:
            if not self.hour_results.loading or self.hour_results.error:
                return False
            progress=small_json(self.training/'progress.json',2_000_000) or self.resume
            self.seat_results.update(progress.get('completed',0),self.options.get('seed'),self.plan.get('hashes',{}).get('bin/paisho-gen5'))
            self.hour_results.update(self.seat_results.offset, self.options.get('seed'),
                                     self.plan.get('hashes',{}).get('bin/paisho-gen5'))
            return self.hour_results.loading and self.hour_results.error is None

    def psr(self, game_id):
        if not re.fullmatch(r'\d{1,10}',game_id):raise FileNotFoundError()
        path=self.training/'games'/f'game-{int(game_id):07}.psr'
        if path.is_symlink():raise FileNotFoundError()
        with path.open('rb') as f:data=f.read(2_000_001)
        if len(data)>2_000_000:raise ValueError('oversized PSR')
        return data


def page(token=""):
    # Reuse the original panel's exact palette, typography and cards.
    if __package__:
        from .paisho_control import render_control_page
    else:
        from paisho_control import render_control_page
    old=render_control_page(SimpleNamespace(name='Gen5',control_token='',dashboard=None)).decode()
    style=old.split('<style>',1)[1].split('</style>',1)[0]
    return ('''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pai Sho · Gen5</title><style>'''+style+'''
.training-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}button:disabled{opacity:.45;cursor:default}select{background:var(--bg);color:var(--ink);border:1px solid var(--line);padding:8px;border-radius:6px}.period-switch{display:flex;gap:8px;margin-bottom:12px}.period-switch button[aria-pressed="true"]{border-color:var(--jade);color:var(--jade)}.header-actions{display:flex;gap:18px;align-items:center;flex-wrap:wrap;width:100%}.header-actions>.training-controls,.header-actions>.gen3-controls{width:100%;padding:8px 0;border-top:1px solid var(--line)}#elo-metrics strong{color:var(--gold)}a{color:var(--jade)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);font-size:.85rem}.scroll{overflow:auto}h2{font-family:Georgia,serif}.metric strong{overflow-wrap:anywhere}svg{width:100%;height:110px}.tag{color:var(--gold)}
</style><main><header><div><p class="eyebrow">PAI SHO · CONTRÔLE LOCAL</p><h1>Génération 5</h1></div><div class="header-actions"><div class="training-controls"><strong>Auto-jeu et rejeu Gen5</strong><button id="train-stop" onclick="trainingAction('stop')">Stop</button><button id="train-pause" onclick="trainingAction('pause')">Pause</button><button id="train-resume" onclick="trainingAction('resume')">Reprendre</button><button class="train-duration" onclick="trainingAction('start',2,'selfplay')">2 h</button><button class="train-duration" onclick="trainingAction('start',6,'selfplay')">6 h</button><button class="train-duration" onclick="trainingAction('start',12,'selfplay')">12 h</button><button class="train-duration" onclick="trainingAction('start',24,'selfplay')">24 h</button><label for="train-hours">Heures</label><input id="train-hours" type="number" min="0.016667" max="24" step="any" value="6" style="width:65px"><button class="train-duration" onclick="trainingAction('start',Number(document.getElementById('train-hours').value),'selfplay')">Démarrer</button></div><div class="training-controls legacy-controls"><strong>Rejeu contre Gen3.1</strong><button id="legacy-stop" onclick="trainingAction('stop')">Stop</button><button id="legacy-pause" onclick="trainingAction('pause')">Pause</button><button id="legacy-resume" onclick="trainingAction('resume',undefined,'gen3-replay')">Reprendre</button><button class="legacy-duration" onclick="trainingAction('start',2,'gen3-replay')">2 h</button><button class="legacy-duration" onclick="trainingAction('start',6,'gen3-replay')">6 h</button><button class="legacy-duration" onclick="trainingAction('start',12,'gen3-replay')">12 h</button><button class="legacy-duration" onclick="trainingAction('start',24,'gen3-replay')">24 h</button><label for="legacy-hours">Heures</label><input id="legacy-hours" type="number" min="0.016667" max="24" step="any" value="6" style="width:65px"><button class="legacy-duration" onclick="trainingAction('start',Number(document.getElementById('legacy-hours').value),'gen3-replay')">Démarrer</button></div><div class="gen3-controls" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><strong id="gen3-title">Entraîner Gen3.2</strong><button id="gen3-stop" onclick="gen3Action('stop')">Stop</button><button id="gen3-pause" onclick="gen3Action('pause')">Pause</button><button id="gen3-resume" onclick="gen3Action('resume')">Reprendre</button><button class="gen3-duration" onclick="gen3Action('start',2)">2 h</button><button class="gen3-duration" onclick="gen3Action('start',6)">6 h</button><button class="gen3-duration" onclick="gen3Action('start',12)">12 h</button><input id="gen3-hours" type="number" min="0.016667" max="24" step="any" value="1" aria-label="Gen3 training hours" style="width:65px"><button class="gen3-duration" onclick="gen3Action('start',Number(document.getElementById('gen3-hours').value))">Démarrer</button><button id="gen3-evaluate" onclick="gen3Action('evaluate')">Comparer à Gen3.1</button><button id="gen3-next" onclick="gen3Action('next')">Version suivante</button></div><button id="language" type="button" aria-label="Switch to English">English</button><a href="/examples">Parties récentes · PSR</a></div></header><label for="dashboard-view">Tableau affiché</label> <select id="dashboard-view" onchange="selectDashboard(this.value)"><option value="auto">Campagne active</option><option value="gen3">Gen3.2</option><option value="gen5">Gen5</option></select><p id="gen3-status" class="note"></p><div id="gen3-comparisons" class="scroll"></div><div id="gen3-games" class="scroll"></div><details class="panel"><summary>Protocole Gen3.1 → Gen3.x</summary><div id="gen3-protocol"></div></details><p id="ladder-status" class="note"></p><p id="training-action-status" class="note"></p><p class="note">Pause conserve la RAM et le temps restant. Reprise directe si disponible ; sinon, reprise après sauvegarde. Choisissez une durée quand la campagne est en pause ou terminée.</p><p class="note">Stop annule les heures restantes et conserve une sauvegarde reprenable. Les deux modes poursuivent le même modèle Gen5, un seul à la fois.</p>
<section class="panel"><div class="metrics" id="main"></div><p id="state" class="note">Lecture de la campagne…</p><progress id="time" max="100" value="0"></progress></section>
<section class="panel"><h2>Évaluations et Elo interne</h2><div class="metrics" id="elo-metrics"></div><p id="elo">En attente d’une évaluation.</p><p id="evaluation-progress" class="note"></p><p class="note">Écarts provisoires contre le modèle initial, séparés des résultats contre Gen3.1. Les checkpoints de poids sont conservés régulièrement ; les jalons de +100 Elo demandent une évaluation complète.</p></section>
<section class="panel"><h2>Production et apprentissage</h2><div class="metrics" id="learning"></div><p class="note" id="recipe"></p><svg id="spark" viewBox="0 0 1000 110" role="img" aria-label="Débit Gen5 · fenêtre glissante de dix minutes"></svg><p class="note">Débit calculé avec les compteurs natifs, réanalyses exclues. Fenêtre observée de dix minutes maximum ; moyenne de la session en attendant deux relevés.</p></section>
<section class="panel"><h2>Répartition des parties</h2><div class="period-switch" role="group"><button id="period-total" type="button" aria-pressed="true" onclick="setPeriod('total')">Total</button><button id="period-hour" type="button" aria-pressed="false" onclick="setPeriod('hour')">Dernière heure</button></div><p id="period-note" class="note"></p><div class="scroll"><table><thead><tr><th>File</th><th>Terminées / tentatives</th><th>Gen5 gagne</th><th>Adversaire gagne</th><th>Nulles</th><th>Décisions</th><th>Positions neuves apprises</th><th id="replay-heading">Relectures</th></tr></thead><tbody id="lanes"></tbody></table></div><p class="note">Host et Guest sont des places tirées au sort. Les victoires des confrontations sont attribuées au modèle, quelle que soit sa place. En auto-jeu, deux Gen5 jouent : chaque partie décisive donne une victoire et une défaite au même modèle.</p></section>
<section class="panel"><h2 id="history-heading">Évolution heure par heure</h2><p id="history-description" class="note">Tranches horaires fixes · heure de Paris · résultats enregistrés, reprises comprises. L’heure en cours est partielle.</p><label for="hourly-lane">File</label> <select id="hourly-lane" onchange="if(latest)render(latest)"><option value="Historical">Gen5 contre Gen3.1</option><option value="Selfplay">Auto-jeu Gen5</option><option value="Checkpoint">Gen5 · versions retenues</option></select><p id="hourly-status" class="note"></p><div class="scroll"><table><thead><tr><th>Heure</th><th>Terminées / tentatives</th><th>Gen5 gagne</th><th>Adversaire gagne</th><th>Nulles</th><th>Taux de victoire</th><th>Positions neuves apprises</th></tr></thead><tbody id="hourly-games"></tbody></table></div><p class="note">Taux de victoire calculé parmi les parties terminées de la tranche. Les interruptions restent visibles dans les tentatives.</p></section>
<section class="panel"><h2>Historique Elo par heure</h2><p class="note">Dernière mesure publiée dans chaque heure, sans moyenne ni report de l’heure précédente. Écarts provisoires face à chaque référence fixe ; Gen5 initiale vaut 0 Elo. Les lots et limites peuvent différer : les résultats résolus et indéterminés restent indiqués.</p><p id="elo-timeline-status" class="note"></p><div class="scroll"><table><thead><tr><th>Heure</th><th>Référence</th><th>Elo mesuré</th><th>Mesuré à</th><th>Version évaluée</th><th>V / N / D</th><th>Indéterminées</th><th>Simulations / limite de décisions</th></tr></thead><tbody id="hourly-elo"></tbody></table></div></section>

<section class="panel" id="examples"><h2>Dernières parties sélectionnées</h2><p class="note">Quelques parties récentes, courtes, longues ou nulles. Actualisation toutes les 30 secondes. Toutes les parties originales sont conservées dans la campagne.</p><div class="scroll"><table><thead><tr><th>Partie</th><th>Modèle joué</th><th>File</th><th>Place de Gen5</th><th>Résultat</th><th>Décisions</th><th>Secondes</th><th>PSR</th></tr></thead><tbody id="games"></tbody></table></div><p id="copy-status" class="note"></p></section><p class="note" id="error"></p></main>
<script>const controlToken = '''+json.dumps(token)+''';\n'''+Path(__file__).with_name('paisho_dashboard_lanes.js').read_text()+'\n'+Path(__file__).with_suffix('.js').read_text()+'\n'+Path(__file__).with_name('paisho_gen3_dashboard.js').read_text()+'\n'+Path(__file__).with_name('paisho_gen3_panels.js').read_text()+'''</script></html>''').encode()


def make_handler(reader):
    controls=getattr(reader,'actions',None)
    token=small_json(controls.path).get('token','') if controls else ''
    if __package__:
        from .paisho_gen3_controls import Gen3Controls
        from .paisho_gen3_reader import Gen3Reader
    else:
        from paisho_gen3_controls import Gen3Controls
        from paisho_gen3_reader import Gen3Reader
    gen3=Gen3Controls(gen5=controls) if controls else None
    gen3_reader=Gen3Reader(gen3) if gen3 else None
    action_lock=threading.RLock()
    content=page(token)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in ('/api/control','/api/gen3/control') or controls is None:
                self.send_error(404);return
            expected=f'127.0.0.1:{self.server.server_port}'
            if (self.headers.get('Host')!=expected or self.headers.get('Origin',f'http://{expected}')!=f'http://{expected}'
                or not secrets.compare_digest(self.headers.get('X-Paisho-Control',''),token)):
                self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1024:raise ValueError('Requête invalide.')
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict):raise ValueError("Requête invalide.")
                with action_lock:
                    if self.path=='/api/gen3/control':result=gen3.request(request.get('action'),request.get('hours'))
                    else:
                        if request.get('action') in ('start','resume') and gen3.active():raise ValueError('Pause or finish Gen3 before starting Gen5.')
                        result=controls.request(request.get('action'),request.get('hours'),**({'mode':request['mode']} if request.get('mode') is not None else {}))
                data=json.dumps(result).encode();code=200
            except (OSError,ValueError,TypeError) as exc:
                data=json.dumps({'error':str(exc)}).encode();code=409
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path in ('/','/examples','/examples/','/dashboard'):data=content;kind='text/html; charset=utf-8';download=None
                elif (g:=re.fullmatch(r'/gen3/games/(\d{1,10})\.psr',path)) and gen3 is not None:data=gen3.psr(g[1]);kind='text/plain; charset=utf-8';download=f'gen3-game-{g[1]}.psr'
                elif (g:=re.fullmatch(r'/gen3/selected/([0-9a-f]{16}-\d{1,10})\.psr',path)) and gen3_reader is not None:data=gen3_reader.psr(g[1]);kind='text/plain; charset=utf-8';download=f'gen3-{g[1]}.psr'
                elif path=='/api/gen3/status' and gen3 is not None:data=json.dumps(gen3_reader.read()).encode();kind='application/json';download=None
                elif path=='/api/status':data=json.dumps(reader.read()).encode();kind='application/json';download=None
                elif (match:=re.fullmatch(r'/examples/files/(\d{1,10})\.psr',path)):
                    data=reader.psr(match[1]);kind='text/plain; charset=utf-8';download=f'gen5-game-{match[1]}.psr'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
                if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
                self.end_headers();self.wfile.write(data)
            except FileNotFoundError:self.send_error(404)
            except (OSError,ValueError,TypeError) as e:self.send_error(503,str(e))
        def log_message(self,*args):pass
    return Handler


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args(argv)
    reader=ControlSession(a.campaign,Gen5Reader)
    warmup=time.process_time()
    for _ in range(16):
        reader.current().next_status=0
        current=reader.read()
        if reader.current().seat_results.processed==current['progress'].get('completed',0) or time.process_time()-warmup>=.05:
            break
    server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(reader))
    # Hydrate each newly selected campaign gradually, even with a hidden page.
    # Once caught up this only checks the active pointer/loading flag.
    stop_warmup=threading.Event()
    def warm_history():
        while not stop_warmup.wait(1):
            reader.warm_history()
    threading.Thread(target=warm_history,name='dashboard-history-load',daemon=True).start()
    print(f'Gen5 interface: http://127.0.0.1:{a.port}/',flush=True)
    try:server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:pass
    finally:
        stop_warmup.set()
        server.server_close()

if __name__=='__main__':main()
