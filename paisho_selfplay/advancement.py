"""Small durable 20-minute outcome archives, written during existing telemetry."""
import errno
import json
from pathlib import Path
import time
from paisho_ppo import budget,storage
from .league import ADVANCEMENT_SECONDS


def archive(directory,population,include_current=False,now=None):
    now=time.time() if now is None else now
    current=int(now//ADVANCEMENT_SECONDS)*ADVANCEMENT_SECONDS
    root=Path(directory)/'mcts-advancement';written=0
    intervals=population.get('advancement_intervals',{})
    for stamp,rows in list(intervals.items()):
        start=int(stamp);closed=start<current
        if not closed and not include_current:continue
        value={'schema':1,'start_unix':start,'end_unix':start+ADVANCEMENT_SECONDS,
            'interval_seconds':ADVANCEMENT_SECONDS,'closed':closed,'generations':rows}
        path=root/(stamp+'.json')
        # Closed buckets are immutable: no recurring writes or disk scans.
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('closed'):
                intervals.pop(stamp,None)
                continue
            if previous==value:continue
        size=len(json.dumps(value).encode())
        try:
            with budget.project_write_guard():
                budget.check(reserve=2*size+budget.METADATA_RESERVE)
                root.mkdir(exist_ok=True)
                storage._atomic_json(path,value)
            written+=1
            if closed:intervals.pop(stamp,None)
        except budget.SpaceUnavailable:
            return written
        except OSError as error:
            if error.errno not in (errno.ENOSPC,errno.EDQUOT):raise
            return written
    return written
