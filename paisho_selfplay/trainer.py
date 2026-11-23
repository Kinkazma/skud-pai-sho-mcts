"""Same learned Transformer and terminal PPO, with a separate run and flow."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import signal
import time
import torch
from paisho_ppo import storage
from paisho_ppo.execution import Execution
from paisho_ppo.trainer import Trainer
from paisho_ppo.config import Config
from paisho_ppo.engine import engine_identity
from .identity import source_identity
from .metadata import write as write_measurement
from . import stream
from .reward import LEGACY,PROFILES


def runtime():
    path=Path(__file__).resolve().parents[1]/'selfplay/runtime-profile.json'
    return json.loads(path.read_text()) if path.exists() else {
        'engine_workers':4,'cpu_threads':2,'execution':{
            'adam_foreach':True,'clip_foreach':False,'async_transfer':True}}


class SelfplayTrainer(Trainer):
    def __init__(self,config,engine,device):
        super().__init__(config,engine,device)
        self.runtime=runtime()
        torch.set_num_threads(self.runtime['cpu_threads'])
        self.execution_options=self.runtime['execution']
        self.source_identity=source_identity()
        self.finished_times=deque()
        self.telemetry_seconds=0.0
        self.reward_profile=LEGACY
        self.population=None
        self.population_rng=torch.Generator().manual_seed(config.seed+4)
        self.opponent_pool=None
        self.flow_decisions={'selfplay':0,'mcts':0}

    @classmethod
    def restore(cls,directory,engine,device,allow_seed_upgrade=False):
        saved,metadata,recovered=storage.load(directory)
        recorded=saved.get('selfplay_source_identity')
        if recorded and recorded!=source_identity():
            is_seed=saved['counters']['optimizer_steps']==0 and saved['pending'] is None
            if not (allow_seed_upgrade and is_seed):
                raise ValueError('Selfplay source differs from this checkpoint; trained weights require explicit migration')
        if saved['engine_identity']!=engine_identity():
            raise ValueError('Checkpoint uses different rules source')
        if saved['source_identity']!=storage.source_identity():
            raise ValueError('Shared human-training source differs from checkpoint')
        if saved['points']!=engine.description['points']:
            raise ValueError('Checkpoint board encoding differs')
        trainer=cls(Config.from_dict(saved['config']),engine,device)
        trainer.model.load_state_dict(saved['model'],strict=True)
        trainer.optimizer.load_state_dict(saved['optimizer'])
        trainer.sampling_rng.set_state(saved['sampling_rng'])
        trainer.shuffle_rng.set_state(saved['shuffle_rng'])
        trainer.reservoir_rng.set_state(saved['reservoir_rng'])
        torch.set_rng_state(saved['torch_rng'])
        trainer.baseline,trainer.pending=saved['baseline'],saved['pending']
        trainer.counters,trainer.history=saved['counters'],saved['history']
        trainer.counters.setdefault('length_outlier_stops',0)
        trainer.flow_decisions=saved.get('flow_decisions',{'selfplay':0,'mcts':0})
        trainer.behavior={v:{k:t.to(trainer.device) for k,t in p.items()}
            for v,p in saved.get('behavior',{}).items()}
        trainer.ready=saved.get('ready',[])
        trainer.outliers=saved.get('outliers',trainer.outliers)
        trainer.training_stage,trainer.pretraining=saved['training_stage'],saved['pretraining']
        trainer.telemetry_seconds=saved.get('telemetry_seconds',0.0)
        trainer.reward_profile=saved.get('reward_profile',LEGACY)
        if trainer.reward_profile not in PROFILES:raise ValueError('Unknown terminal reward profile')
        if saved.get('population') is not None:
            from .league import Population
            from .opponents import Pool,identity
            if saved['population'].get('opponent_identity')!=identity():
                raise ValueError('Published opponents or resident search executable changed')
            trainer.population_rng.set_state(saved['population_rng'])
            trainer.population=Population(saved['population'],trainer.population_rng)
            trainer.opponent_pool=Pool(trainer.config.environments,trainer.runtime['engine_workers'])
        if trainer.pending is not None:
            observations=engine.request('restore',games=saved['games'])
            if observations!=trainer.pending['observations']:
                raise ValueError('Restored games differ from checkpoint observations')
            if any(o['rules']!=trainer.config.rules for o in observations):
                raise ValueError('Restored rules differ from configuration')
            if trainer.opponent_pool is not None:
                trainer.opponent_pool.restore(trainer.pending['matches'],saved['games'])
        if recovered:
            write_measurement(Path(directory)/'state.json',metadata)
            print(json.dumps({'recovered_previous_checkpoint':metadata['generation']}),flush=True)
        return trainer

    def record_closed_games(self,previous_done):
        if self.population is not None:
            pending=self.pending
            for slot,(before,after) in enumerate(zip(previous_done,pending['done'])):
                if before or not after:continue
                outcome=pending['outcomes'][slot];termination=pending['terminations'][slot]
                if outcome!=-2:
                    self.counters['finished_games']+=1
                    self.counters[{-1:'draws',0:'host_wins',1:'guest_wins'}[outcome]]+=1
                elif termination=='move-limit-unknown':self.counters['length_outlier_stops']+=1
                else:self.counters['blocked_stops']+=1
                self.population.record(pending['matches'][slot],outcome,pending['decisions'][slot],termination)
            return
        before=self.counters['finished_games']
        super().record_closed_games(previous_done)
        stamp=time.monotonic()
        self.finished_times.extend([stamp]*(self.counters['finished_games']-before))
        while self.finished_times and self.finished_times[0]<stamp-30:self.finished_times.popleft()

    def payload(self):
        if self.opponent_pool is not None and self.pending is not None:
            self.opponent_pool.drain(self.pending['matches'])
        result=super().payload()
        result['selfplay_source_identity']=self.source_identity
        result['selfplay_runtime']=self.runtime
        result['telemetry_seconds']=self.telemetry_seconds
        result['reward_profile']=self.reward_profile
        result['flow_decisions']=self.flow_decisions
        if self.population is not None:
            result['population']=self.population.state
            result['population_rng']=self.population_rng.get_state()
        return result

    def summary(self):
        result=super().summary();stamp=time.monotonic()
        while self.finished_times and self.finished_times[0]<stamp-30:self.finished_times.popleft()
        result.update(kind='empty-board-terminal-selfplay',selfplay_source_identity=self.source_identity,
            finished_games_last_30_seconds=len(self.finished_times),
            finished_games_per_second_30s=len(self.finished_times)/30,
            game_move_limit=None,game_time_limit=None,loop_observations_do_not_stop_games=True,
            selfplay_runtime=self.runtime,telemetry_seconds=self.telemetry_seconds)
        result['reward_profile']=self.reward_profile
        result['decisions_by_source']=dict(self.flow_decisions)
        result['behavior_refresh']='per-decision' if self.runtime.get('live_behavior',False) else 'per-game'
        if self.population is not None:
            from .league import LIMIT
            league=self.population.summary()
            result.update(population=league,game_move_limit=LIMIT,
                finished_games_last_30_seconds=league['rates']['selfplay']['completed_30s'],
                finished_games_per_second_30s=league['rates']['selfplay']['per_second_30s'],
                mcts_finished_games_last_30_seconds=league['rates']['mcts']['completed_30s'],
                mcts_finished_games_per_second_30s=league['rates']['mcts']['per_second_30s'])
        return result

    def run(self,directory,seconds,max_rollouts=None):
        self.start_ppo()
        write_measurement(Path(directory)/'progress.json',{
            'measured_at':datetime.now(timezone.utc).isoformat(),**self.summary()})
        deadline=time.monotonic()+seconds;last_save=last_progress=time.monotonic()
        last_control=0.0;control_id=None
        self._clock=time.monotonic();initial=self.counters['rollouts'];stop_requested=False
        saver=None
        if self.runtime.get('async_checkpoints',False):
            from .checkpoints import PeriodicSaver
            saver=PeriodicSaver()
        def stop(signum,frame):
            nonlocal stop_requested
            stop_requested=True
        old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        for s in old:signal.signal(s,stop)
        try:
            while time.monotonic()<deadline and not stop_requested:
                if saver is not None:saver.poll(self)
                if time.monotonic()-last_control>=1:
                    last_control=time.monotonic()
                    path=Path(directory)/'control.json'
                    try: control=json.loads(path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError): control={}
                    if control.get('id')!=control_id:
                        control_id=control.get('id')
                        if control.get('action')=='stop':stop_requested=True;continue
                        if control.get('action')=='timer':
                            remaining=datetime.fromisoformat(control['until']).timestamp()-time.time()
                            deadline=time.monotonic()+max(0,remaining)
                if max_rollouts is not None and self.counters['rollouts']-initial>=max_rollouts:break
                before=self.counters['rollouts']
                metrics=stream.tick(self,deadline,lambda:stop_requested,prefetch=True)
                if self.counters['rollouts']!=before:
                    print(json.dumps(self.history[-1],allow_nan=False),flush=True)
                if metrics is not None and metrics.get('budget_stop'):break
                if time.monotonic()-last_save>=self.config.checkpoint_seconds:
                    if saver is None:self.save(directory)
                    else:saver.submit(self,directory)
                    last_save=time.monotonic()
                # Metadata must not wait on the checkpoint's storage lock.
                # The next idle writer moment publishes all current counters.
                if time.monotonic()-last_progress>=30 and (saver is None or saver.future is None):
                    begin=time.perf_counter()
                    write_measurement(Path(directory)/'progress.json',{
                        'measured_at':datetime.now(timezone.utc).isoformat(),**self.summary()})
                    if self.population is not None:
                        from .advancement import archive
                        archive(directory,self.population.state)
                    self.telemetry_seconds+=time.perf_counter()-begin
                    last_progress=time.monotonic()
        finally:
            try:
                if saver is not None:saver.close(self)
            finally:
                self.account_time();self._clock=None;self.save(directory)
            write_measurement(Path(directory)/'progress.json',{
                'measured_at':datetime.now(timezone.utc).isoformat(),**self.summary()})
            if self.population is not None:
                from .advancement import archive
                archive(directory,self.population.state,include_current=True)
            for s,handler in old.items():signal.signal(s,handler)
            if self.opponent_pool is not None:self.opponent_pool.close();self.opponent_pool=None
        print(json.dumps(self.summary(),allow_nan=False),flush=True)
