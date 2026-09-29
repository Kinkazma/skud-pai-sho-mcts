"""Explicit Gen3 lineage controller; independent weights and time from Gen5."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import uuid
if __package__:
    from .paisho_gen5_campaign import write
    from .paisho_gen5_actions import read, native
else:
    from paisho_gen5_campaign import write
    from paisho_gen5_actions import read, native
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'training-runs/gen3-lineage/control.json'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Gen3Controls:
    def __init__(self,state=DEFAULT,gen5=None):self.path=Path(state);self.gen5=gen5
    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            yield
    def snapshot(self):
        s=read(self.path)
        if not s:return {'state':'unavailable','generation':'3.2'}
        result={k:s.get(k) for k in ('state','generation','remaining_seconds','error','campaign','started','end','candidate','assessment','versions')}
        if s.get('campaign'):
            p=read(Path(s['campaign'])/'training/progress.json');result['progress']=p
        if s.get('assessment'):
            result['comparisons']=[{k:r.get(k) for k in ('budget','wins','draws','losses','unknown','conditional_internal_elo')} for r in read(s['assessment']).get('reports',[])]
        result['ready']=s.get('ready',False)
        return result
    def install(self,binary,model,dataset):
        with self.locked():
            s=read(self.path)
            if s.get('state') in ('starting','running','pausing','stopping','fitting','evaluating'):raise ValueError('Gen3 is active')
            if s:raise ValueError('Gen3 lineage already installed')
            artifact=read(model)
            write(self.path,{'state':'ready','generation':artifact['generation'],'ready':True,'binary':str(Path(binary).resolve()),'binary_sha256':digest(binary),'model':str(Path(model).resolve()),'model_sha256':digest(model),'dataset':str(Path(dataset).resolve()),'dataset_sha256':digest(dataset),'remaining_seconds':0,'gen5_control':str(self.gen5.path) if self.gen5 else None})
    def upgrade_training(self,binary,reference,*,tactics=None):
        """Explicit paused transition; retain learned state and remaining time."""
        with self.locked():
            s=read(self.path)
            if s.get('state')!='paused':raise ValueError('Pause Gen3 before changing the training protocol')
            if tactics is not None:
                if (set(tactics) != {'solver','tactical_positions','correction_replay'}
                        or type(tactics['solver']) is not bool or type(tactics['correction_replay']) is not bool
                        or type(tactics['tactical_positions']) is not int or not 0 <= tactics['tactical_positions'] <= 100000):
                    raise ValueError('Invalid Gen3 tactical protocol')
                s['training_options']=dict(tactics)
            s.update(binary=str(Path(binary).resolve()),binary_sha256=digest(binary),historical_reference=str(Path(reference).resolve()),historical_reference_sha256=digest(reference))
            write(self.path,s)
            return self.snapshot()
    def stage_candidate(self,binary,model,pool):
        """Explicit new candidate; archive previous paused lineage including its time/FIFO."""
        binary=Path(binary).resolve();model=Path(model).resolve()
        info=json.loads(subprocess.check_output([str(binary),'inspect',str(model)],text=True))
        if info.get('generation')!='3.4' or info.get('value_features')!=128:
            raise ValueError('Expected verified Gen3.4 value128 candidate')
        if {e.get('generation') for e in pool}!={'Gen3.1','Gen3.2','Gen3.3'}:
            raise ValueError('Expected frozen Gen3.1, Gen3.2 and Gen3.3 experts')
        for e in pool:
            if digest(e['path'])!=e['sha256']:raise ValueError('Historical expert changed')
        with self.locked():
            old=read(self.path)
            if old.get('state') not in ('paused','stopped','completed','ready'):
                raise ValueError('Finish or pause Gen3 before staging a candidate')
            if old.get('pid'):raise ValueError('Previous native process has not drained')
            archive=self.path.parent/'lineage-states'/('gen'+old['generation']+'-'+uuid.uuid4().hex[:12]+'.json')
            archive.parent.mkdir(parents=True,exist_ok=True)
            write(archive,old)
            versions=dict(old.get('versions',{}));versions[old['generation']]={'control_archive':str(archive),'campaign':old.get('campaign'),'model':old['model'],'remaining_seconds':old.get('remaining_seconds',0)}
            state={k:old[k] for k in ('dataset','dataset_sha256','gen5_control','replay_index','training_options') if k in old}
            state.update(state='ready',ready=True,generation='3.4',binary=str(binary),binary_sha256=digest(binary),model=str(model),model_sha256=digest(model),remaining_seconds=0,campaign=None,candidate=None,versions=versions,historical_pool=pool,post_training_fit=False,evaluation_anchor=str(model),previous_control=str(archive),protocol='GEN3_4_VALUE_MEMORY_V2',resource_options={'replay_max_bytes':48*1024**3})
            write(self.path,state)
            return self.snapshot()

    def stage_residual_generation(self,binary,model,pool,checkpoint):
        """Explicit Gen3.4 checkpoint -> Gen3.5; preserve its replay, archive old control."""
        binary=Path(binary).resolve();model=Path(model).resolve();checkpoint=Path(checkpoint).resolve()
        source=read(checkpoint);parent=read(source['model']);artifact=read(model)
        info=json.loads(subprocess.check_output([str(binary),'inspect',str(model)],text=True))
        if (info.get('generation')!='3.5' or info.get('value_features')!=128
                or info.get('value_residual_parameters')!=2081 or parent.get('generation')!='3.4'
                or artifact.get('parent_sha256')!=digest(source['model'])):
            raise ValueError('Expected neutral Gen3.5 migration of the selected Gen3.4 checkpoint')
        for key in ('value128_extra','memory_scope','memory_manifest','memory_manifest_sha256','updates'):
            if artifact.get(key)!=parent.get(key):raise ValueError('Migration changed '+key)
        if (artifact['compact']['weights']!=parent['compact']['weights']
                or artifact['policy']['parameters']!=parent['policy']['parameters']
                or artifact['policy'].get('sequence_memory')!=parent['policy'].get('sequence_memory')
                or any(artifact['value_residual'][2064:])):
            raise ValueError('Migration must preserve weights and start with a zero residual output')
        replay=Path(source['replay_index']).resolve()
        if not replay.is_file():raise ValueError('Selected checkpoint replay is missing')
        expected={(g,b) for g in ('Gen3.1','Gen3.2','Gen3.3','Gen3.4') for b in (32,64,128,256,512)}
        if len(pool)!=len(expected) or {(e['generation'],e['budget']) for e in pool}!=expected:
            raise ValueError('Expected all four frozen generations at each configured budget')
        for e in pool:
            if digest(e['path'])!=e['sha256']:raise ValueError('Historical expert changed')
            if e['generation']=='Gen3.4' and e['sha256']!=digest(source['model']):
                raise ValueError('Gen3.4 reference must be the selected parent checkpoint')
        with self.locked():
            old=read(self.path)
            if old.get('state') not in ('paused','stopped','completed','ready') or old.get('pid'):
                raise ValueError('Finish or pause and drain Gen3 before staging Gen3.5')
            if old.get('generation')!='3.4':raise ValueError('Gen3.5 transition already applied or wrong lineage')
            archive=self.path.parent/'lineage-states'/('gen3.4-'+uuid.uuid4().hex[:12]+'.json')
            archive.parent.mkdir(parents=True,exist_ok=True)
            write(archive,old)
            state={k:old[k] for k in ('dataset','dataset_sha256','gen5_control','training_options','resource_options') if k in old}
            versions=dict(old.get('versions',{}));versions['3.4']={'control_archive':str(archive),'model':old['model'],'campaign':old.get('campaign'),'remaining_seconds':old.get('remaining_seconds',0)}
            state.update(state='ready',ready=True,generation='3.5',binary=str(binary),binary_sha256=digest(binary),model=str(model),model_sha256=digest(model),replay_index=str(replay),replay_index_sha256=digest(replay),remaining_seconds=0,campaign=None,candidate=None,versions=versions,historical_pool=pool,historical_every=5,post_training_fit=False,evaluation_anchor=str(model),previous_control=str(archive),source_checkpoint=str(checkpoint),protocol='GEN3_5_RESIDUAL_80_20_V1')
            write(self.path,state)
            return self.snapshot()

    def set_historical_every(self,every):
        """Change the reference schedule only after a durable pause."""
        if type(every) is not int or every < 2:raise ValueError('Invalid historical interval')
        with self.locked():
            s=read(self.path)
            if s.get('state')!='paused' or s.get('pid'):raise ValueError('Pause and drain Gen3 first')
            if not s.get('historical_pool'):raise ValueError('Expected a frozen historical pool')
            s['historical_every']=every
            if s.get('generation')=='3.4':s['protocol']='GEN3_4_REFERENCE_MIX_V3'
            write(self.path,s)
        return self.snapshot()

    def psr(self,game):
        if not str(game).isdigit():raise ValueError('Invalid game id')
        s=read(self.path)
        if not s.get('campaign'):raise FileNotFoundError('No Gen3 game')
        return (Path(s['campaign'])/'training/games'/f'game-{int(game):07}.psr').read_bytes()
    def active(self):return read(self.path).get('state') in ('starting','running','pausing','stopping','fitting','evaluating')
    def request(self,action,hours=None):
        with self.locked():
            s=read(self.path)
            if not s.get('ready'):raise ValueError('Gen3 continuation is not ready')
            active=s.get('state') in ('starting','running','pausing','stopping','fitting','evaluating')
            if action in ('pause','stop'):
                if not active:raise ValueError('No active Gen3 campaign')
                if s['state']!='running' and not (action=='stop' and s['state'] in ('fitting','evaluating')):raise ValueError('Wait for the current operation to finish')
                remaining=max(0,s['end']-time.time()) if action=='pause' else 0
                out=Path(s['campaign'])/'training'
                out.mkdir(parents=True,exist_ok=True)
                write(out/('pause-request.json' if action=='pause' else 'stop-request.json'),{'time':time.time()})
                s.update(state='pausing' if action=='pause' else 'stopping',remaining_seconds=remaining)
                write(self.path,s);return self.snapshot()
            if active:raise ValueError('Gen3 is already active')
            if action=='next':
                if s['state'] not in ('completed','stopped') or not s.get('candidate'):raise ValueError('Complete the current version first')
                version=s['generation'].split('.');generation=f'{version[0]}.{int(version[1])+1}'
                artifact=read(s['model']);artifact['generation']=generation
                path=self.path.parent/f'initial-{generation}-{uuid.uuid4().hex[:8]}.json';write(path,artifact)
                history=s.setdefault('versions',{});history[s['generation']]={k:s.get(k) for k in ('candidate','campaign','assessment')}
                s.update(campaign=None,model=str(path),model_sha256=digest(path),generation=generation,state='ready',remaining_seconds=0,candidate=None,assessment=None)
                write(self.path,s);return self.snapshot()
            if action not in ('start','resume','evaluate'):raise ValueError('Unknown Gen3 action')
            if self.gen5:
                other=read(self.gen5.path);campaign=Path(other['campaign'])
                if other.get('busy') or native(campaign) and not other.get('paused'):raise ValueError('Pause or finish Gen5 before starting Gen3 on the ten CPU cores')
            if action=='evaluate':
                if not s.get('candidate'):raise ValueError('No completed Gen3 candidate')
                seconds=0
            else:
                seconds=float(hours)*3600 if action=='start' else float(s.get('remaining_seconds',0))
                if not math.isfinite(seconds) or not (60 if action=='start' else 1)<=seconds<=86400:raise ValueError('Duration must be between one minute and 24 hours')
            for key in ('binary','model','dataset')+ (('historical_reference',) if s.get('historical_reference') else ()):
                if digest(s[key])!=s[key+'_sha256']:raise ValueError(f'{key} changed')
            for expert in s.get('historical_pool',[]):
                if digest(expert['path'])!=expert['sha256']:raise ValueError('Historical expert changed')
            s.update(state='evaluating' if action=='evaluate' else 'starting',error=None,remaining_seconds=seconds,action_id=uuid.uuid4().hex)
            write(self.path,s)
            try:
                with (self.path.parent/'worker.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--state',str(self.path),'--id',s['action_id'],'--operation',action],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                s['worker_pid']=child.pid;write(self.path,s)
                threading.Thread(target=child.wait,daemon=True).start()
            except Exception as e:
                s.update(state='failed',error=str(e));write(self.path,s);raise
            return self.snapshot()

def auxiliary(command_args,control,log=None):
    if read(control.path).get('state')=='stopping':return False
    child=subprocess.Popen(command_args,stdout=log,stderr=subprocess.STDOUT if log else None)
    try:
        while child.poll() is None:
            if read(control.path).get('state')=='stopping':
                child.terminate();child.wait();return False
            time.sleep(.1)
        if child.returncode:raise ValueError(f'Auxiliary process failed: {child.returncode}')
        return read(control.path).get('state')!='stopping'
    finally:
        if child.poll() is None:child.terminate();child.wait()

def stopped(control):
    with control.locked():
        current=read(control.path);current.update(state='stopped',remaining_seconds=0,pid=None);write(control.path,current)

def worker(path,identity,operation):
    control=Gen3Controls(path)
    with control.locked():
        s=read(path)
        if s.get('action_id')!=identity:raise ValueError('Superseded request')
    out=path.parent/('gen'+s['generation']+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+identity[:6])
    child=None;awake=None
    try:
        out.mkdir();(out/'bin').mkdir();shutil.copy2(s['binary'],out/'bin/paisho-gen32')
        binary=out/'bin/paisho-gen32'
        if digest(binary)!=s['binary_sha256']:raise ValueError('Frozen binary mismatch')
        if operation=='evaluate':
            with control.locked():
                current=read(path);current.update(evaluation_output=str(out),evaluation_generation=s['generation']);write(path,current)
            parent=ROOT/'benchmarks/results/gen3-rules-v2-training-2026-09-09/final/human-model.json'
            reports=[]
            for budget in (32,64,128,256,512):
                destination=out/f'budget-{budget}'
                if not auxiliary([str(binary),'compare',s['model'],str(parent),str(destination),str(budget),'16'],control):stopped(control);return
                reports.append(read(destination/'report.json'))
            write(out/'assessment.json',{'reports':reports,'promoted':False,'site_elo':None,'generation':s['generation'],'published_unix_seconds':time.time()})
            with control.locked():
                current=read(path);history=current.setdefault('assessment_history',[]);history.append(str(out/'assessment.json'));current.update(state='completed',assessment=str(out/'assessment.json'));write(path,current)
            return
        shutil.copy2(s['model'],out/'initial-model.json');shutil.copy2(s['dataset'],out/'human-dataset.json')
        if digest(out/'initial-model.json')!=s['model_sha256'] or digest(out/'human-dataset.json')!=s['dataset_sha256']:raise ValueError('Frozen model/dataset mismatch')
        seconds=s['remaining_seconds']
        options=json.loads(subprocess.check_output([str(binary),'defaults'],text=True))
        options.update(model=str(out/'initial-model.json'),output=str(out/'training'),seconds=seconds,threads=10,actors=10,archive_workers=2,seed=32001+int(identity[:8],16))
        if s.get('historical_reference'):
            reference=out/'historical-gen3-1.json';shutil.copy2(s['historical_reference'],reference)
            if digest(reference)!=s['historical_reference_sha256']:raise ValueError('Frozen Gen3.1 mismatch')
            options.update(historical_reference=str(reference),historical_reference_sha256=s['historical_reference_sha256'],historical_every=s.get('historical_every',10))
        if s.get('historical_pool'):
            frozen=[]
            for expert in s['historical_pool']:
                reference_path=out/('reference-'+expert['sha256']+'.json')
                if not reference_path.exists():shutil.copy2(expert['path'],reference_path)
                if digest(reference_path)!=expert['sha256']:raise ValueError('Frozen expert mismatch')
                frozen.append(dict(expert,path=str(reference_path)))
            options.update(historical_pool=frozen,historical_reference=None,historical_reference_sha256=None,historical_every=s.get('historical_every',10))
        if s.get('resource_options'):options.update(s['resource_options'])
        if s.get('training_options'):options.update(s['training_options'])
        if s.get('replay_index'):options['replay_index']=s['replay_index']
        write(out/'config.json',options)
        write(out/'plan.json',{'protocol':s.get('protocol') or ('GEN3_TACTICAL_LEARNING_V1' if s.get('training_options') else ('GEN3_CONTINUATION_V3' if s.get('historical_reference') else 'GEN3_CONTINUATION_V2')),'options':options,'generation':s['generation'],'parent_sha256':s['model_sha256'],'binary_sha256':s['binary_sha256'],'dataset_sha256':s['dataset_sha256'],'automatic_resume':False})
        with (out/'training.log').open('wb') as log:
            child=subprocess.Popen([str(binary),'run',str(out/'config.json')],stdout=log,stderr=subprocess.STDOUT)
            awake=subprocess.Popen(['caffeinate','-i','-w',str(child.pid)]) if sys.platform=='darwin' else None
            with control.locked():
                current=read(path);history=current.setdefault('campaign_history',[]);history.extend(p for p in [current.get('campaign'),str(out)] if p and p not in history);current.update(state='running',campaign=str(out),pid=child.pid,started=time.time(),end=time.time()+seconds);write(path,current)
            code=child.wait()
            if awake and awake.poll() is None:awake.terminate()
        if code:raise ValueError(f'Native Gen3 exited with code {code}; see {out}/training.log')
        checkpoint=read(out/'training/checkpoint.json')
        if not checkpoint.get('model') or not checkpoint.get('replay_index'):raise ValueError('Missing durable checkpoint')
        with control.locked():
            current=read(path);paused=current['state']=='pausing';was_stopped=current['state']=='stopping'
            current.update(model=checkpoint['model'],model_sha256=digest(checkpoint['model']),replay_index=checkpoint['replay_index'],pid=None)
            if paused or was_stopped:
                current.update(state='paused' if paused else 'stopped',candidate=None if paused else checkpoint['model']);write(path,current);return
            if s.get('post_training_fit',True) is False:
                current.update(state='completed',remaining_seconds=0,candidate=checkpoint['model']);write(path,current);return
            current.update(state='fitting',remaining_seconds=0);write(path,current)
        # Original Gen3.1 post-training human value fit; no inference benchmark.
        with (out/'human-fit.log').open('wb') as log:
            if not auxiliary([str(binary),'human-fit',checkpoint['model'],str(out/'human-dataset.json'),str(out/'human-fit')],control,log):stopped(control);return
        candidate=out/'human-fit/model.json'
        with control.locked():
            current=read(path);current.update(state='completed',remaining_seconds=0,model=str(candidate),model_sha256=digest(candidate),candidate=str(candidate));write(path,current)
    except Exception as e:
        with control.locked():
            current=read(path);current.update(state='failed',error=str(e));write(path,current)
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        if awake is not None and awake.poll() is None:awake.terminate()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['worker']);p.add_argument('--state',type=Path,required=True);p.add_argument('--id',required=True);p.add_argument('--operation',required=True);a=p.parse_args();worker(a.state,a.id,a.operation)
