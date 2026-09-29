"""Drain an identified paused native before recovering its latest durable state."""
import os
import signal
import time


def drain(campaign,pid,read,native,write,timeout=180,resident_pause=None,save_pause=None):
    if native(campaign)!=pid:raise ValueError('Native identity changed before RAM capture.')
    if __package__:
        from .paisho_gen5_resident import exclude_pause,pause_ticket,stopped
    else:
        from paisho_gen5_resident import exclude_pause,pause_ticket,stopped
    # Validate and exclude the suspension BEFORE any signal or stop request.
    offset=exclude_pause(campaign,pid,resident_pause,read,native)
    resident=read(campaign/'training/progress.json')
    write(campaign/'training/stop-request.json',{'requested_unix_seconds':time.time()})
    continued=False
    try:
        os.kill(pid,signal.SIGCONT)
        continued=True
        limit=time.monotonic()+timeout
        while time.monotonic()<limit:
            current=native(campaign)
            if current is None:break
            if current!=pid:raise ValueError('Native identity changed during RAM capture.')
            time.sleep(.1)
        else:raise ValueError('RAM capture is still pending; no replacement was launched.')
        while read(campaign/'status.json').get('state') in ('starting','running') and time.monotonic()<limit:
            time.sleep(.1)
        if read(campaign/'status.json').get('returncode')!=0:
            raise ValueError('Native RAM capture did not complete successfully; saved files are preserved.')
        progress=read(campaign/'training/durable-progress.json')
        if not all(k in progress for k in ('version','updates','completed')):
            raise ValueError('Final durable progress is missing after RAM capture.')
        if any(progress[k]<resident.get(k,0) for k in ('version','updates','completed')):
            raise ValueError('Final durable state is older than resident progress; no replacement was launched.')
        return {'pid':pid,'kind':'cooperative-native-drain',
                'paused_time_preserved':offset is not None,'pause_offset_ns':offset,
                **{k:progress[k] for k in ('version','updates','completed')}}
    except Exception:
        # A timeout must not leave an unbounded active campaign or destroy RAM.
        if continued and native(campaign)==pid:
            pause_ns=time.monotonic_ns()
            os.kill(pid,signal.SIGSTOP)
            for _ in range(100):
                if stopped(pid):break
                time.sleep(.01)
            else:raise ValueError('RAM capture suspension is not confirmed; no replacement was launched.')
            # The old ticket would also exclude the ACTIVE drain on retry.
            # Rebase on this suspension and the already-applied clock offset.
            ticket=pause_ticket(campaign,pid,read,pause_ns)
            if save_pause is not None:save_pause(ticket)
        raise
