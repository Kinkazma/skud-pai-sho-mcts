"""Native cumulative-counter throughput, independent of journal backfill and UI history.

Use up to ten minutes of observed progress. Before that window exists, label the
exact current-session average explicitly. No games, model reads or probes run here.
"""
from collections import deque


class CounterRates:
    def __init__(self):
        self.samples = deque()
        self.active_origin = None

    def observe(self, elapsed, counters, initial_elapsed=0., initial_counters=None, *,
                exclude_loading=False, active_start=None):
        if elapsed is None or not counters:
            return dict(ready=False, seconds=0, terminal_per_second={}, terminal={}, timestamp_basis='native-counters')
        elapsed=float(elapsed); initial_elapsed=float(initial_elapsed or 0)
        counts={k:int(v) for k,v in counters.items()}
        initial={k:int((initial_counters or {}).get(k,0)) for k in counts}
        if ((self.samples and elapsed < self.samples[-1][0]) or
                (self.active_origin is not None and elapsed < self.active_origin[0])):
            self.samples.clear()
            self.active_origin = None
        if exclude_loading:
            if active_start is not None:
                initial_elapsed = max(initial_elapsed, float(active_start))
            else:
                # Old runtimes have no exact loading-end marker. Start from the
                # first observed increase, never from inherited totals or zeros.
                if self.active_origin is None:
                    if not any(v > initial.get(k, 0) for k, v in counts.items()):
                        return dict(ready=False, seconds=0, terminal_per_second={}, terminal={},
                                    timestamp_basis='awaiting-computation')
                    self.active_origin = (elapsed, counts.copy())
                initial_elapsed, initial = self.active_origin
            if elapsed <= initial_elapsed:
                return dict(ready=False, seconds=0, terminal_per_second={}, terminal={},
                            timestamp_basis='awaiting-computation')
        if not self.samples or elapsed>self.samples[-1][0]:
            self.samples.append((elapsed,counts))
        while len(self.samples)>1 and elapsed-self.samples[0][0]>600:
            self.samples.popleft()
        if exclude_loading and not self.samples[0][0] == initial_elapsed and 0 < elapsed-initial_elapsed <= 600:
            self.samples.appendleft((initial_elapsed, initial.copy()))
        if len(self.samples)>1:
            start,base=self.samples[0]
            basis='native-counters-observed-window'
        else:
            start,base=self.samples[0] if exclude_loading else (initial_elapsed,initial)
            basis='native-counters-observed-window' if exclude_loading else 'native-counters-session-average'
        # Include the known session origin while the entire session fits the window.
        if 0<elapsed-initial_elapsed<=600:
            start,base,basis=initial_elapsed,initial,'native-counters-session-window'
        if exclude_loading:
            basis += '-excluding-loading'
        seconds=max(0.,elapsed-start)
        terminal={k:max(0,v-base.get(k,0)) for k,v in counts.items()}
        return dict(ready=seconds>0,seconds=seconds,maximum_seconds=600 if basis!='native-counters-session-average' else None,
                    timestamp_basis=basis,terminal=terminal,
                    terminal_per_second={k:v/seconds if seconds>0 else None for k,v in terminal.items()})
