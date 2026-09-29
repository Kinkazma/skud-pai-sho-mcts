"""Verified same-process resume; the native mapped clock preserves paused time.

Old binaries remain eligible only for an explicitly fixed deadline. A changed
binary/protocol or exited process still needs the ordinary durable handoff.
"""
import hashlib
from pathlib import Path
import os
import signal
import struct
import subprocess
import time


def stopped(pid):
    return 'T' in subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True)


def clock_info(campaign,pid,read):
    info=read(campaign/'training/resident-clock.json')
    if info.get('schema')!='paisho-resident-clock-v1' or info.get('pid')!=pid:return None
    path=campaign/'training.pause-clock.bin'
    if Path(info.get('path','')).resolve()!=path.resolve():return None
    with path.open('rb') as f:
        stat=os.fstat(f.fileno());data=f.read(9)
    if len(data)!=8:return None
    return dict(path=str(path),pid=pid,device=stat.st_dev,inode=stat.st_ino,offset_ns=struct.unpack('<Q',data)[0])


def pause_ticket(campaign,pid,read,monotonic_ns):
    info=clock_info(campaign,pid,read)
    return dict(info,monotonic_ns=monotonic_ns) if info else None


def exclude_pause(campaign,pid,ticket,read,native):
    """Apply a verified pause before either resident resume or cooperative drain.

    Always derive the offset from the original ticket, so a failed SIGCONT may
    be retried without counting the pause twice. This helper never sends signals.
    """
    info=clock_info(campaign,pid,read)
    if info is None:
        if ticket:raise ValueError('Resident clock disappeared; process remains paused.')
        return None  # Compatibility with old natives without a mapped clock.
    if not ticket:raise ValueError('Resident pause ticket missing; process remains paused.')
    if any(ticket.get(k)!=info[k] for k in ('path','pid','device','inode')):
        raise ValueError('Resident clock identity changed; process remains paused.')
    now=time.monotonic_ns()
    if now<ticket['monotonic_ns'] or info['offset_ns']<ticket['offset_ns']:
        raise ValueError('Invalid resident pause clock; process remains paused.')
    offset=ticket['offset_ns']+now-ticket['monotonic_ns']
    with open(ticket['path'],'r+b',buffering=0) as f:
        stat=os.fstat(f.fileno());data=f.read(9)
        if (stat.st_dev,stat.st_ino)!=(ticket['device'],ticket['inode']) or len(data)!=8:
            raise ValueError('Resident clock file changed; process remains paused.')
        current=struct.unpack('<Q',data)[0]
        if not ticket['offset_ns']<=current<=offset:
            raise ValueError('Resident clock offset changed; process remains paused.')
        if native(campaign)!=pid or not stopped(pid):
            raise ValueError('Resident process is no longer suspended.')
        f.seek(0)
        if f.write(struct.pack('<Q',offset))!=8:raise OSError('Incomplete resident clock update')
        os.fsync(f.fileno())
    return offset


def eligible(state,campaign,mode,read,native):
    if not state.get('paused') or state.get('prepared_next'):return None
    plan=read(campaign/'plan.json')
    options=read(campaign/'config.json')
    old_mode='gen3-replay' if options.get('legacy_replay') else 'selfplay'
    if mode is not None and mode!=old_mode:return None
    pid=native(campaign)
    if not pid or not stopped(pid):return None
    expected=plan.get('hashes',{}).get('bin/paisho-gen5')
    if not expected or hashlib.sha256((campaign/'bin/paisho-gen5').read_bytes()).hexdigest()!=expected:return None
    if (state.get('runtime') or {}).get('sha256',expected)!=expected:return None
    if hashlib.sha256((campaign/'config.json').read_bytes()).hexdigest()!=plan.get('hashes',{}).get('config.json'):return None
    deadline=plan.get('end_unix_seconds',0)
    if state.get('resume_policy')=='fixed-deadline':
        clock=clock_info(campaign,pid,read)
        if clock and clock['offset_ns']!=0:return None
        return dict(pid=pid,deadline=deadline,clock=None) if deadline>time.time() else None
    ticket=state.get('resident_pause');info=clock_info(campaign,pid,read)
    if not ticket or not info or state.get('remaining_seconds',0)<=0:return None
    if any(ticket.get(k)!=info[k] for k in ('path','pid','device','inode')):return None
    if info['offset_ns']<ticket['offset_ns'] or time.monotonic_ns()<ticket['monotonic_ns']:return None
    return dict(pid=pid,deadline=deadline,clock=ticket)


def resume(state,campaign,mode,read,native,save):
    match=eligible(state,campaign,mode,read,native)
    if match is None:return False
    pid=match['pid'];deadline=match['deadline'];ticket=match['clock']
    if ticket:
        offset=exclude_pause(campaign,pid,ticket,read,native)
        deadline+=offset/1e9
        status=read(campaign/'status.json');status['end_unix_seconds']=deadline
        # Preserve coordinator-owned fields and its final status merge.
        try:from .paisho_gen5_campaign import write
        except ImportError:from paisho_gen5_campaign import write
        write(campaign/'status.json',status)
    state.update(paused=False,stopped=False,remaining_seconds=max(0,deadline-time.time()),error=None,
                 last_resume={'kind':'resident','pid':pid,'deadline':deadline,'replay_reload':False,
                              'paused_time_preserved':bool(ticket)})
    save(state)
    try:os.kill(pid,signal.SIGCONT)
    except OSError:
        state.update(paused=True);save(state);raise
    return True
