"""Explicit local UI actions for frozen Gen5 campaigns; no automatic restart."""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

if __package__:
    from .paisho_gen5_campaign import start, write
    from .prepare_gen5_resume import prepare
    from .paisho_gen5_modes import apply as apply_mode, MODES
    from .paisho_gen5_capture import drain as capture_native
else:
    from paisho_gen5_campaign import start, write
    from prepare_gen5_resume import prepare
    from paisho_gen5_modes import apply as apply_mode, MODES
    from paisho_gen5_capture import drain as capture_native


def read(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {}


def command(pid):
    if not pid:return ''
    value=subprocess.run(['ps','-p',str(int(pid)),'-o','stat=','-o','command='],capture_output=True,text=True).stdout.strip().split(None,1)
    # During exit macOS may already have discarded argv while the PID remains
    # visible as (paisho-gen5). It is no longer an actionable native command.
    return value[1] if len(value)==2 and 'Z' not in value[0] and value[1]!='(paisho-gen5)' else ''


def native(campaign):
    status=read(campaign/'status.json');pid=status.get('training_pid')
    actual=command(pid)
    if not actual:return None
    expected=f'{campaign}/bin/paisho-gen5 run {campaign}/config.json'
    if actual!=expected:
        # macOS can briefly replace argv with an exit label before reap.
        time.sleep(.02)
        actual=command(pid)
        if not actual:return None
        if actual==expected:return pid
        if status.get('state') in ('completed','failed'):return None
        raise ValueError(f'Identité du processus différente ; aucune action effectuée. Attendu : {expected!r} ; observé : {actual!r}')
    return pid


def duration(value):
    hours=float(value)
    if not math.isfinite(hours) or not 1/60<=hours<=24:
        raise ValueError('Durée : de 1 minute à 24 heures.')
    return hours


class Actions:
    def __init__(self,campaign,state_path=None):
        campaign=Path(campaign).resolve()
        pointer=read(campaign/'dashboard-control-root.json')
        self.path=Path(state_path or pointer.get('state') or campaign/'dashboard-control.json')
        self.lock=threading.RLock()
        with self.locked():
            if not self.path.exists():
                self.save({'campaign':str(campaign),'paused':False,'token':secrets.token_urlsafe(32)})

    @contextmanager
    def locked(self):
        with self.lock, self.path.with_suffix('.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)

    def save(self,state):write(self.path,state)

    def install_runtime(self,binary,archive_workers=1):
        """Enable validated recipes for future clicks; never alter a running process."""
        binary=Path(binary).resolve()
        with self.locked():
            state=read(self.path)
            if state.get('busy'):raise ValueError('An action is in progress.')
            state['runtime']={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'archive_workers':archive_workers}
            self.save(state)

    def stage(self, config, binary):
        """Select the next explicit launch; never start a process."""
        with self.locked():
            state=read(self.path);campaign=Path(state['campaign'])
            if (native(campaign) and not state.get('paused')) or state.get('busy') or (not state.get('paused') and read(campaign/'status.json').get('state')!='completed'):
                raise ValueError('Prepare a new protocol only after the current campaign completes or is paused.')
            options=read(config);progress=read(campaign/'training/durable-progress.json') or read(campaign/'training/progress.json')
            resumed=read(options.get('resume_progress','missing'))
            if not options.get('case_curriculum') or resumed.get('version')!=progress.get('version') or resumed.get('updates')!=progress.get('updates'):
                raise ValueError('Prepared curriculum must preserve the latest durable model.')
            paths={'config':str(Path(config).resolve()),'binary':str(Path(binary).resolve())}
            state['prepared_next']={**paths,'hashes':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()},
                                    'campaign':str(campaign),'version':progress['version'],'updates':progress['updates'],
                                    'protocol':{'structural_repair':bool(options.get('structural_repair')),'recall_fraction':options.get('recall_fraction'),
                                                'learning_loop_repair':bool(options.get('learning_loop_repair')),
                                                'learning_loop_v2':bool(options.get('learning_loop_v2')),
                                                'neural_memory':bool(options.get('neural_memory')),
                                                'value_policy_strength':options.get('value_policy_strength',0)}}
            state['prepared_next']['protocol']['minimum_search_depth']=options.get('minimum_search_depth',0)
            self.save(state)
        return self.snapshot()

    def _snapshot(self):
        state=read(self.path)
        job=state.get('job_pid')
        busy=bool(job and f'{Path(__file__).resolve()} --worker {self.path}' in command(job))
        if state.get('busy') and not busy:
            pending=state.get('pending_campaign')
            # A worker may have launched successfully immediately before a
            # dashboard/worker interruption. Never launch another copy.
            if pending and (Path(pending)/'started.json').exists():
                state.update(campaign=pending,paused=False)
            state.update(busy=False,error='Action interrompue ; état récupéré, aucune relance automatique.')
            self.save(state)
        campaign=Path(state['campaign']);pid=native(campaign)
        return {**{k:state.get(k) for k in ('campaign','paused','remaining_seconds','error','prepared_next','stopped','busy_action')},
                'mode':'gen3-replay' if read(campaign/'config.json').get('legacy_replay') else 'selfplay',
                'modes_available':bool(state.get('runtime')),
                'busy':busy,'running':bool(pid) and not state.get('paused'),
                'native_pid':pid,'memory_resident':bool(pid),
                'resident_resume_available':bool(pid and state.get('paused') and state.get('resident_pause') and not state.get('prepared_next')
                    and (state.get('runtime') or {}).get('sha256') in (None,read(campaign/'plan.json').get('hashes',{}).get('bin/paisho-gen5'))),
                'last_resume':state.get('last_resume')}

    def snapshot(self):
        with self.locked():return self._snapshot()

    def request(self,action,hours=None,mode=None):
        with self.locked():
            status=self._snapshot();state=read(self.path)
            if status['busy']:raise ValueError('Une action est déjà en cours.')
            campaign=Path(state['campaign']);pid=native(campaign)
            if mode is not None and mode not in MODES:raise ValueError('Unknown training mode.')
            if mode is not None and not state.get('runtime'):raise ValueError('Training modes are not ready.')
            if action=='stop':
                if state.get('stopped'):return status
                if not pid and not state.get('paused'):raise ValueError('Aucun entraînement à arrêter.')
                state.update(busy=True,error=None,pending_campaign=None,busy_action=action)
                with self.path.with_suffix('.log').open('ab') as log:
                    worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(self.path),'--stop'],stdout=log,stderr=log,start_new_session=True)
                state['job_pid']=worker.pid;self.save(state)
            elif action=='pause':
                if state.get('paused'):return status
                if not pid:raise ValueError('Aucun entraînement actif à mettre en pause.')
                remaining=max(0,read(campaign/'status.json').get('end_unix_seconds',time.time())-time.time())
                pause_ns=time.monotonic_ns()
                os.kill(pid,signal.SIGSTOP)
                if __package__:
                    from .paisho_gen5_resident import pause_ticket, stopped
                else:
                    from paisho_gen5_resident import pause_ticket, stopped
                state.update(paused=True,stopped=False,remaining_seconds=remaining,error=None,resident_pause=None)
                try:
                    for _ in range(100):
                        if stopped(pid):break
                        time.sleep(.01)
                    else:raise ValueError('La suspension du processus n’est pas confirmée.')
                    state['resident_pause']=pause_ticket(campaign,pid,read,pause_ns)
                except (OSError, ValueError) as exc:
                    # Keep the real suspension visible even if optional resident
                    # metadata cannot be read. Never restart to hide this error.
                    state['error']=str(exc)
                    self.save(state)
                    raise
                try:self.save(state)
                except OSError:
                    if native(campaign)==pid:os.kill(pid,signal.SIGCONT)
                    raise
            elif action in ('resume','start'):
                if pid and not state.get('paused'):
                    raise ValueError('Mettez la campagne en pause avant de choisir une nouvelle durée.')
                if action=='resume':
                    if __package__:
                        from .paisho_gen5_resident import resume as resume_resident
                    else:
                        from paisho_gen5_resident import resume as resume_resident
                    if resume_resident(state,campaign,mode,read,native,self.save):
                        return self._snapshot()
                if action=='resume':
                    seconds=state.get('remaining_seconds',0)
                    if seconds<=0:raise ValueError('Choisissez une nouvelle durée.')
                    hours=seconds/3600
                hours=duration(hours)
                if action=='resume' and mode is not None and mode!=status['mode']:
                    raise ValueError('Reprendre conserve le mode ; utilisez Stop puis une nouvelle durée.')
                # Write the intent before launching a detached worker. It survives
                # closing the tab or restarting the dashboard process.
                state.update(busy=True,error=None,pending_campaign=None,busy_action=action)
                with self.path.with_suffix('.log').open('ab') as log:
                    worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(self.path)]+(['--mode',mode] if mode else [])+['--hours',str(hours)],
                        stdout=log,stderr=log,start_new_session=True)
                state['job_pid']=worker.pid;self.save(state)
            else:raise ValueError('Action inconnue.')
        return self.snapshot()

    def _capture_paused(self,campaign,pid,state):
        def save_pause(ticket):
            with self.locked():
                latest=read(self.path)
                latest.update(paused=True,resident_pause=ticket)
                self.save(latest)
        captured=capture_native(campaign,pid,read,native,write,
                                resident_pause=state.get('resident_pause'),save_pause=save_pause)
        # Failure preserves the original ticket, or the new timeout ticket.
        with self.locked():
            latest=read(self.path);latest.pop('resident_pause',None)
            latest['last_capture']=captured;self.save(latest)
        return captured

    def continue_run(self,hours,mode=None):
        with self.locked():
            state=read(self.path);campaign=Path(state['campaign'])
            if state.get('job_pid')!=os.getpid():raise ValueError('Action remplacée.')
        try:
            hours=duration(hours);pid=native(campaign)
            if pid and not state.get('paused'):raise ValueError('La campagne doit être en pause.')
            if pid:
                process_state=subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True)
                if 'T' not in process_state:raise ValueError('Le processus n’est pas suspendu.')
            work=campaign.parent/('gen5-ui-recovery-'+uuid.uuid4().hex[:12])
            prepared=state.get('prepared_next')
            binary=campaign/'bin/paisho-gen5'
            if prepared:
                if prepared['campaign']!=str(campaign):raise ValueError('Prepared protocol belongs to another campaign.')
                for key in ('config','binary'):
                    if hashlib.sha256(Path(prepared[key]).read_bytes()).hexdigest()!=prepared['hashes'][key]:
                        raise ValueError('Prepared '+key+' changed; no campaign launched.')
                current=read(campaign/'training/durable-progress.json') or read(campaign/'training/progress.json')
                if any(current[k]!=prepared[k] for k in ('version','updates')):raise ValueError('Prepared model is stale.')
                options=read(prepared['config']);binary=Path(prepared['binary'])
            else:options=read(campaign/'config.json')
            old_mode='gen3-replay' if options.get('legacy_replay') else 'selfplay'
            runtime=state.get('runtime')
            if runtime and not prepared:
                binary=Path(runtime['binary'])
                if hashlib.sha256(binary.read_bytes()).hexdigest()!=runtime['sha256']:raise ValueError('Installed runtime changed.')
                apply_mode(options,mode or old_mode)
                options['archive_workers']=runtime.get('archive_workers',1)
            elif prepared and mode is not None:
                apply_mode(options,mode)
            elif mode is not None:raise ValueError('No validated runtime for this mode.')
            if pid and read(campaign/'config.json').get('control_stop'):
                # Recovery must follow the final native commit, not a checkpoint
                # which predates the learner still held by the suspended process.
                captured=self._capture_paused(campaign,pid,state)
                pid=None
                if prepared and any(captured[k]!=prepared[k] for k in ('version','updates')):
                    raise ValueError('La RAM a été sauvegardée ; la préparation doit être actualisée sur ce dernier modèle avant reprise.')
            progress=read(campaign/'training/progress.json')
            prior=read(options.get('resume_progress',campaign/'missing-resume.json'))
            if prepared or not progress or progress.get('completed',0)==prior.get('completed',0):
                # Pause during initial replay import: the untouched frozen inputs
                # are already the exact durable starting point.
                # A staged migration also owns its verified model/FIFO. Re-running
                # recovery from a newer non-durable progress view would discard it.
                if not options.get('resume_progress') or not options.get('replay_index'):
                    raise ValueError('Aucune sauvegarde durable disponible.')
                work.mkdir()
                shutil.copy2(options['resume_progress'],work/'resume-progress.json')
                shutil.copy2(options['replay_index'],work/'replay.index.json')
                receipt={'model':options['model']}
            else:
                prepare(campaign,work)
                receipt=read(work/'receipt.json')
            if mode is not None and mode!=old_mode:
                resumed=read(work/'resume-progress.json')
                # Keep weights, FIFO, fixed anchor and ladder; pending case turns belong to the old opponent.
                resumed['case_states']={}
                write(work/'resume-progress.json',resumed)
            options.update(model=receipt['model'],resume_progress=str(work/'resume-progress.json'),
                           replay_index=str(work/'replay.index.json'))
            write(work/'config.json',options)
            out=campaign.parent/('micro-gen5-ui-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            # Durable recovery is verified before releasing the old process/RAM.
            if pid:
                if native(campaign)!=pid:raise ValueError('Identité du processus modifiée.')
                os.kill(pid,signal.SIGKILL)
                for _ in range(100):
                    if not command(pid):break
                    time.sleep(.05)
                else:raise ValueError('Ancien processus encore présent ; aucune nouvelle campagne lancée.')
            with self.locked():
                state=read(self.path);state['pending_campaign']=str(out);self.save(state)
            end=dt.datetime.fromtimestamp(time.time()+hours*3600,dt.timezone.utc).isoformat()
            result=start(work/'config.json',binary,out,end,history_source=campaign/'training/history')
            write(out/'dashboard-control-root.json',{'state':str(self.path)})
            with self.locked():
                state=read(self.path);state.pop('prepared_next',None);state.update(campaign=str(out),paused=False,stopped=False,busy=False,
                    job_pid=None,pending_campaign=None,remaining_seconds=hours*3600,error=None,last_start=result)
                self.save(state)
        except Exception as exc:
            with self.locked():
                state=read(self.path)
                pending=state.get('pending_campaign')
                if pending and (Path(pending)/'started.json').exists():state.update(campaign=pending,paused=False)
                state.update(busy=False,job_pid=None,error=str(exc));self.save(state)
            raise


    def stop_run(self):
        with self.locked():
            state=read(self.path);campaign=Path(state['campaign'])
            if state.get('job_pid')!=os.getpid():raise ValueError('Action remplacée.')
        try:
            pid=native(campaign);options=read(campaign/'config.json')
            if pid and options.get('control_stop') and state.get('paused'):
                self._capture_paused(campaign,pid,state)
            elif pid and options.get('control_stop'):
                write(campaign/'training/stop-request.json',{'requested_unix_seconds':time.time()})
                limit=time.monotonic()+180
                while native(campaign) and time.monotonic()<limit:time.sleep(.1)
                if native(campaign):raise ValueError('Arrêt toujours en cours ; aucune nouvelle campagne lancée.')
                # Coordinator publishes the final return code after the native exits.
                while read(campaign/'status.json').get('state') in ('running','starting') and time.monotonic()<limit:time.sleep(.1)
                if read(campaign/'status.json').get('returncode')!=0:raise ValueError('Le moteur a signalé une erreur pendant l’arrêt.')
            elif pid:
                # Compatibility with already frozen V16 executables: capture first.
                if not state.get('paused'):os.kill(pid,signal.SIGSTOP)
                with self.locked():
                    current=read(self.path);current['paused']=True;current['remaining_seconds']=max(0,read(campaign/'status.json').get('end_unix_seconds',time.time())-time.time());self.save(current)
                if 'T' not in subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True):raise ValueError('Pause non confirmée.')
                capture=campaign.parent/('gen5-stop-recovery-'+uuid.uuid4().hex[:12])
                prepare(campaign,capture)
                if native(campaign)!=pid:raise ValueError('Identité modifiée.')
                os.kill(pid,signal.SIGKILL)
                limit=time.monotonic()+30
                while native(campaign) and time.monotonic()<limit:time.sleep(.1)
                while read(campaign/'status.json').get('state') in ('running','starting') and time.monotonic()<limit:time.sleep(.1)
                if native(campaign):raise ValueError('Ancien processus encore actif.')
                status=read(campaign/'status.json')
                status.update(state='completed',termination='user-stop',controlled_stop=True,stop_recovery=str(capture))
                write(campaign/'status.json',status)
            with self.locked():
                state=read(self.path);state.pop('prepared_next',None);state.pop('resident_pause',None)
                state.update(stopped=True,paused=False,remaining_seconds=0,busy=False,job_pid=None,error=None)
                self.save(state)
        except Exception as exc:
            with self.locked():
                state=read(self.path);state.update(busy=False,job_pid=None,error=str(exc));self.save(state)
            raise


class ControlSession:
    def __init__(self,campaign,factory):
        self.actions=Actions(campaign);self.factory=factory;self.reader=None;self.lock=threading.RLock()

    def current(self):
        with self.lock:
            campaign=Path(read(self.actions.path)['campaign'])
            if self.reader is None or self.reader.campaign!=campaign:
                self.reader=self.factory(campaign)
            return self.reader

    def read(self):
        status=self.actions.snapshot()
        result={**self.current().read(),'controls':status}
        if status.get('paused') or status.get('stopped') or not status.get('running') and not status.get('busy'):
            result.update(gen5_terminal_per_second=0.,historical_terminal_per_second=0.)
        return result

    def psr(self,game):return self.current().psr(game)
    def warm_history(self):return self.current().warm_history()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True);parser.add_argument('--hours',type=float);parser.add_argument('--mode',choices=MODES);parser.add_argument('--stop',action='store_true')
    args=parser.parse_args();state=read(args.worker)
    actions=Actions(state['campaign'],args.worker)
    if args.stop:actions.stop_run()
    else:actions.continue_run(args.hours,args.mode)
