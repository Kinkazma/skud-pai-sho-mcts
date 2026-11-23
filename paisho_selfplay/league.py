"""Bounded population curriculum and outcome telemetry; no neural evaluation."""
from collections import deque
import math
import time
import torch

GENERATIONS=tuple(f'3.{i}' for i in range(1,6))
BUDGETS=(1,2,3,4,6,8,16,32,64,128)
RECIPE='resident-gen3-population-700-entropy001-combined-v2'
LIMIT=700
MIN_DECISIVE=100
WINDOW_DECISIVE=200
ADVANCEMENT_SECONDS=1200


def counts():
    return dict(started=0,finished=0,win=0,loss=0,draw=0,aborted=0,blocked=0)


def initial(counters=None,lengths=()):
    counters=counters or {}
    own=counts()
    own.update(started=counters.get('started_games',0),finished=counters.get('finished_games',0),
        draw=counters.get('draws',0),aborted=counters.get('length_outlier_stops',0),
        blocked=counters.get('blocked_stops',0))
    return {'recipe':RECIPE,'launches':0,'deck':[],
        'ladders':{g:{'index':0,'next_player':0,'window':[],'promotions':0} for g in GENERATIONS},
        'counts':{'selfplay':own,'mcts':counts()},'opponents':{g:counts() for g in GENERATIONS},
        'lengths':{'selfplay':list(lengths)[-4096:],'mcts':[]},
        'optimizer_steps':{'selfplay':counters.get('optimizer_steps',0),'mcts':0},'intervals':{},
        'advancement_intervals':{}}


def advancement_interval(state,now):
    bucket=str(int(now//ADVANCEMENT_SECONDS)*ADVANCEMENT_SECONDS)
    intervals=state.setdefault('advancement_intervals',{})
    if bucket not in intervals:
        intervals[bucket]={g:{'budget_start':BUDGETS[l['index']],
            'budget_end':BUDGETS[l['index']],'counts':counts(),'by_budget':{},'promotions':[]}
            for g,l in state['ladders'].items()}
    return intervals[bucket]


class Population:
    def __init__(self,state,rng):
        if state['recipe']!=RECIPE:raise ValueError('Unknown population recipe')
        self.state,self.rng=state,rng
        self.finished_times={'selfplay':deque(),'mcts':deque()}

    def launch(self):
        s=self.state
        if not s['deck']:
            deck=['selfplay']*5+list(GENERATIONS)
            s['deck']=[deck[i] for i in torch.randperm(10,generator=self.rng).tolist()]
        selected=s['deck'].pop();s['launches']+=1
        if selected=='selfplay':match={'kind':'selfplay'}
        else:
            ladder=s['ladders'][selected];player=ladder['next_player'];ladder['next_player']=1-player
            match={'kind':'mcts','generation':selected,'budget':BUDGETS[ladder['index']],
                'transformer_player':player,'seed':int(torch.randint(2**63-1,(1,),generator=self.rng)),
                'choice':None}
            s['opponents'][selected]['started']+=1
        s['counts'][match['kind']]['started']+=1
        return match

    def record(self,match,outcome,length,termination,now=None):
        now=time.time() if now is None else now
        kind=match['kind'];targets=[self.state['counts'][kind]]
        if kind=='mcts':targets.append(self.state['opponents'][match['generation']])
        if outcome==-2:result='aborted' if termination=='move-limit-unknown' else 'blocked'
        elif outcome==-1:result='draw'
        elif kind=='mcts':result='win' if outcome==match['transformer_player'] else 'loss'
        else:result=None
        for target in targets:
            if outcome!=-2:target['finished']+=1
            if result:target[result]+=1
        if outcome!=-2:
            values=self.state['lengths'][kind];values.append(length);del values[:-4096]
            self.finished_times[kind].append(time.monotonic())
        if kind!='mcts':return
        bucket=str(int(now//1800)*1800)
        interval=self.state['intervals'].setdefault(bucket,{g:counts() for g in GENERATIONS})
        row=interval[match['generation']]
        row[result]+=1
        if outcome!=-2:row['finished']+=1
        while len(self.state['intervals'])>96:del self.state['intervals'][min(self.state['intervals'],key=int)]
        ladder=self.state['ladders'][match['generation']]
        advancement=advancement_interval(self.state,now)[match['generation']]
        per_budget=advancement['by_budget'].setdefault(str(match['budget']),counts())
        for target in (advancement['counts'],per_budget):
            target[result]+=1
            if outcome!=-2:target['finished']+=1
        if result in ('win','loss') and match['budget']==BUDGETS[ladder['index']]:
            window=ladder['window']
            window.append(int(result=='win'))
            if len(window)>WINDOW_DECISIVE:
                # Predecessor checkpoints kept the two seats separately and
                # lost their mutual completion order. Retire that whole prefix
                # rather than invent which old outcome to evict first.
                legacy=ladder.pop('legacy_games',0)
                if legacy:del window[:legacy]
                else:del window[:-WINDOW_DECISIVE]
            before=BUDGETS[ladder['index']];games=len(window);wins=sum(window)
            if self.check_promotion(match['generation']):
                advancement['promotions'].append({'at':now,'from_budget':before,
                    'to_budget':BUDGETS[ladder['index']],'decisive_games':games,'wins':wins})
        advancement['budget_end']=BUDGETS[ladder['index']]

    def check_promotion(self,generation):
        ladder=self.state['ladders'][generation];window=ladder['window']
        if len(window)>=MIN_DECISIVE and 2*sum(window)>=len(window) and ladder['index']<len(BUDGETS)-1:
            ladder['index']+=1;ladder['promotions']+=1;ladder['window']=[]
            ladder.pop('legacy_games',None)
            return True
        return False

    def summary(self):
        stamp=time.monotonic();lengths={};rates={}
        for kind in ('selfplay','mcts'):
            events=self.finished_times[kind]
            while events and events[0]<stamp-30:events.popleft()
            rates[kind]={'completed_30s':len(events),'per_second_30s':len(events)/30}
            history=self.state['lengths'][kind]
            ordered=sorted(history[-100:] if kind=='selfplay' else history);n=len(ordered)
            lengths[kind]={'count':n,'median':None if not n else
                (ordered[(n-1)//2]+ordered[n//2])/2,'p90':ordered[math.ceil(.9*n)-1] if n else None,
                'p99':ordered[math.ceil(.99*n)-1] if n else None}
            if kind=='selfplay':lengths[kind].update(window_games=100,history_count=len(history))
        ladders={}
        for g,l in self.state['ladders'].items():
            n=len(l['window']);wins=sum(l['window'])
            ladders[g]={'budget':BUDGETS[l['index']],'decisive_games':n,'wins':wins,
                'win_rate':wins/n if n else None,'promotions':l['promotions']}
        return {'recipe':RECIPE,'launches':self.state['launches'],'counts':self.state['counts'],
            'opponents':self.state['opponents'],'optimizer_steps':self.state['optimizer_steps'],
            'rates':rates,'lengths':lengths,'ladders':ladders,'intervals':self.state['intervals'],
            'interval_seconds':1800,'interval_retention':96,'move_limit':LIMIT}
