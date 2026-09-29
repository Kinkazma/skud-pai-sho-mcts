"""Read-only incremental Gen3 dashboard, independent of models and training IO."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import threading
import time
try:
    from .paisho_throughput import CounterRates
except ImportError:
    from paisho_throughput import CounterRates


def read_json(path, limit=2_000_000):
    try:
        with Path(path).open('rb') as f:
            data=f.read(limit+1)
        if len(data)>limit:raise ValueError('Oversized Gen3 receipt')
        return json.loads(data)
    except FileNotFoundError:return {}


def empty():
    return dict(games=0,terminal=0,wins=0,losses=0,draws=0,decisions=0,fresh=0,replay=0)


def add(total, row):
    for k in total:total[k]+=row[k]


class Gen3Reader:
    def __init__(self, controls):
        self.controls=controls;self.lock=threading.RLock();self.key=None
        self.online={};self.offsets={};self.total={};self.bins={};self.recent=deque();self.selected=deque(maxlen=12)
        self.files={};self.sources={};self.cached=None;self.expires=0
        self.rate_campaign=None;self.counter_rates=CounterRates();self.first_receipt={}

    def read(self, now=None):
        now=time.time() if now is None else now
        with self.lock:
            if self.cached is not None and now<self.expires:return self.cached
            state=read_json(self.controls.path);s=self.controls.snapshot()
            generation=state.get('generation','3.2')
            if generation!=self.key:
                self.key=generation;self.online={};self.offsets={};self.total={};self.bins={};self.recent=deque();self.selected=deque(maxlen=12);self.files={};self.sources={}
            campaigns=list(dict.fromkeys(state.get('campaign_history',[])+([state['campaign']] if state.get('campaign') else [])))
            loading=False;errors=[];active_start=state.get('started',now)
            for campaign in campaigns:
                root=Path(campaign);plan=read_json(root/'plan.json')
                if plan.get('generation',generation)!=generation:continue
                self.sources[hashlib.sha256(campaign.encode()).hexdigest()[:16]]=root
                options=read_json(root/'config.json');ratio=options.get('replay_ratio',4)
                journal=root/'training/receipts.jsonl';offset=self.offsets.get(campaign,0)
                try:
                    with journal.open('rb') as f:
                        f.seek(offset);data=f.read(2_000_000);loading|=f.tell()<journal.stat().st_size
                    end=data.rfind(b'\n')+1
                    for raw in data[:end].splitlines():
                        row=json.loads(raw);offset+=len(raw)+1;self.offsets[campaign]=offset;stamp=row.get('published_unix_seconds')
                        if stamp is None:continue
                        self.first_receipt.setdefault(campaign,stamp)
                        lane=(row['opponent'] if str(row.get('opponent','')).startswith('Gen3.') else 'Historical') if row.get('historical') else 'Selfplay';budget=int(row['budget'])
                        terminal=row.get('termination')=='rules-terminal';outcome=row.get('outcome')
                        learned=int(row.get('learned',0));fresh=learned//(ratio+1);replay=learned-fresh
                        counts=dict(games=1,terminal=int(terminal),wins=int(terminal and outcome=='Win('+str(row.get('candidate_seat'))+')'),losses=int(terminal and outcome not in ('Draw','Win('+str(row.get('candidate_seat'))+')')),draws=int(terminal and outcome=='Draw'),decisions=int(row.get('decisions',0)),fresh=fresh,replay=replay)
                        add(self.total.setdefault(lane,empty()),counts)
                        start=int(stamp//3600)*3600;key=(start,lane,budget)
                        add(self.bins.setdefault(key,empty()),counts)
                        if lane.startswith('Gen3.') and row.get('reference_sha256') and row.get('collector_sha256'):
                            k=(start,budget,lane,row['reference_sha256'],row.get('rules'))
                            v=int(row['collector_version'])
                            online=self.online.setdefault(k,dict(**empty(),version_min=v,version_max=v,published=stamp))
                            for field in empty():online[field]+=counts[field]
                            online['version_min']=min(online['version_min'],v);online['version_max']=max(online['version_max'],v);online['published']=max(online['published'],stamp)
                        if now-stamp<=3600:self.recent.append((stamp,campaign,lane,counts))
                        ident=hashlib.sha256(campaign.encode()).hexdigest()[:16]+'-'+str(row['id'])
                        self.files[ident]=root/'training/games'/f"game-{row['id']:07}.psr"
                        self.selected.append(dict(row,download_id=ident))
                    self.offsets[campaign]=offset
                except FileNotFoundError:pass
                except (ValueError,OSError) as e:errors.append(str(e))
            self.recent=deque(r for r in self.recent if now-r[0]<=3600)
            # Only recent selected files can be fetched, keeping this map bounded.
            ids={r['download_id'] for r in self.selected};self.files={k:v for k,v in self.files.items() if k in ids}
            # A first receipt is a conservative activity origin for old natives.
            # Exclude that receipt from the numerator along with all startup time.
            first_receipt=self.first_receipt.get(state.get('campaign'))
            if first_receipt is not None:active_start=max(active_start,first_receipt)
            hour={};rolling={};window=max(0,min(600,now-active_start))
            for stamp,campaign,lane,row in self.recent:
                add(hour.setdefault(lane,empty()),row)
                if campaign==state.get('campaign') and stamp>active_start and stamp>=now-600:add(rolling.setdefault(lane,empty()),row)
            running=state.get('state')=='running'
            rates={lane:(v['terminal']/window if window>0 and running else 0.) for lane,v in rolling.items()}
            evaluations=[]
            sources=state.get('assessment_history',[])+([state['assessment']] if state.get('assessment') else [])
            for source in dict.fromkeys(sources):
                try:
                    p=Path(source);a=read_json(p);stamp=a.get('published_unix_seconds',p.stat().st_mtime)
                    if a.get('generation',generation)!=generation:continue
                    for report in a.get('reports',[]):
                        evaluations.append({k:report.get(k) for k in ['budget','wins','draws','losses','unknown','conditional_internal_elo','model_sha256','reference_sha256','pairs']}|dict(published=stamp,reference='Gen3.1'))
                except (ValueError,OSError) as e:errors.append(str(e))
            online_evaluations=[]
            for (start,budget,lane,reference,rules),counts in sorted(self.online.items(),reverse=True):
                n=counts['terminal'];w=counts['wins'];d=counts['draws'];l=counts['losses']
                elo=400*math.log10((w+d*.5+1)/(l+d*.5+1)) if n else None
                online_evaluations.append(dict(counts,start=start,budget=budget,reference_sha256=reference,rules=rules,unknown=counts['games']-n,conditional_internal_elo=elo,reference=lane,kind='online-training-cohort'))
            evaluation_progress=None
            if state.get('evaluation_output') and state.get('evaluation_generation')==generation:
                base=Path(state['evaluation_output']);done=resolved=0
                for budget in (32,64,128,256,512):
                    part=base/f'budget-{budget}';report=read_json(part/'report.json')
                    if report:
                        done+=len(report.get('games',[]));resolved+=sum(report.get(k,0) for k in ('wins','draws','losses'))
                        if state.get('state')=='evaluating':
                            evaluations.append({k:report.get(k) for k in ['budget','wins','draws','losses','unknown','conditional_internal_elo','model_sha256','reference_sha256','pairs']}|dict(published=(part/'report.json').stat().st_mtime,reference='Gen3.1'))
                    elif state.get('state')=='evaluating':done+=sum((part/f'game-{i}.psr').exists() for i in range(32))
                evaluation_progress=dict(planned=160,processed=done,resolved=resolved,running=state.get('state')=='evaluating')
            progress=s.get('progress') or {};options=read_json(Path(state['campaign'])/'config.json') if state.get('campaign') else dict(budgets=[512,256,128,64,32,512],historical_every=state.get('historical_every',10),historical_pool=state.get('historical_pool',[]),historical_reference=state.get('historical_reference'),**state.get('resource_options',{}))
            if self.rate_campaign!=state.get('campaign'):
                self.rate_campaign=state.get('campaign');self.counter_rates=CounterRates()
            throughput=self.counter_rates.observe(progress.get('elapsed_seconds'),{'all':progress['terminal']} if 'terminal' in progress else {}, exclude_loading=True,
                active_start=progress.get('computation_start_elapsed_seconds'))
            if not throughput['ready'] and not loading and first_receipt is not None and window>0:
                throughput=dict(ready=True,seconds=window,terminal_per_second={'all':sum(rates.values())},timestamp_basis='receipt-publication-excluding-loading')
            if not running:
                throughput.update(ready=True,terminal_per_second={'all':0.},timestamp_basis='inactive')
            checkpoint=read_json(Path(state['campaign'])/'training/checkpoint.json') if state.get('campaign') else {}
            bins=[dict(start=start,lane=lane,budget=budget,**counts) for (start,lane,budget),counts in sorted(self.bins.items(),reverse=True)]
            s['dashboard']=dict(generation=generation,options=options,checkpoint_updates=checkpoint.get('updates'),totals=self.total,last_hour=hour,bins=bins,bin_seconds=3600,terminal_per_second=throughput["terminal_per_second"].get("all"),lane_rates=rates,window_seconds=throughput["seconds"],throughput_basis=throughput["timestamp_basis"],ready=throughput["ready"],history_ready=not loading,errors=errors,games=list(reversed(self.selected)),evaluations=evaluations,online_evaluations=online_evaluations,evaluation_progress=evaluation_progress,evaluated_versions=len({r['model_sha256'] for r in evaluations}),read_at=now,remaining_seconds=max(0,state.get('end',now)-now) if running else state.get('remaining_seconds',0),progress_age=max(0,now-(Path(state['campaign'])/'training/progress.json').stat().st_mtime) if state.get('campaign') and (Path(state['campaign'])/'training/progress.json').exists() else None)
            self.cached=s;self.expires=now+5;return s

    def psr(self, key):
        with self.lock:
            source,game=key.rsplit('-',1)
            root=self.sources.get(source)
            if root is None or not game.isdigit():raise FileNotFoundError('Gen3 selected game not found')
            path=root/'training/games'/f'game-{int(game):07}.psr'
            return path.read_bytes()
