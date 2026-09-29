"""Rolling receipt-publication results. No training, PSR or model reads.

Existing frozen receipts lack a wall timestamp. Their immutable JSON file mtime
is the publication time, not a guessed game finish time. Bootstrap reads journals
backwards once, aggregates calendar hours, then follows the journal forward.
Each refresh has a small CPU/row budget; incomplete windows stay labelled loading.
"""
import heapq
import json
from pathlib import Path
import time

if __package__:
    from .paisho_gen5_seats import candidate_seat
else:
    from paisho_gen5_seats import candidate_seat

FIELDS = ('games', 'terminal', 'wins', 'losses', 'draws', 'unresolved_seat',
          'decisions', 'fresh_used', 'replay_triggered')


def small(path):
    try:
        with path.open('rb') as stream:
            data = stream.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError('oversized campaign metadata')
        return json.loads(data)
    except FileNotFoundError:
        return {}


def complete_end(path):
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 131073)
            stream.seek(start)
            last = stream.read().rfind(b'\n')
            return start + last + 1 if last >= 0 else 0
    except FileNotFoundError:
        return 0


def reverse_lines(path, end):
    if not end:
        return
    with path.open('rb') as stream:
        carry = b''
        cursor = end
        while cursor:
            start = max(0, cursor - 65536)
            stream.seek(start)
            lines = (stream.read(cursor - start) + carry).split(b'\n')
            carry = lines.pop(0) if start else b''
            for line in reversed(lines):
                if line:
                    if len(line) > 131072:
                        raise ValueError('oversized game receipt')
                    yield line
            if len(carry) > 131072:
                raise ValueError('oversized game receipt')
            cursor = start


class HourResults:
    def __init__(self, campaign):
        self.campaign = Path(campaign).resolve()
        self.offset = complete_end(self.campaign / 'training.log')
        self.bootstrap_end = self.offset
        self.bootstrap = self._old_lines()
        self.loading = True
        self.window_loaded = False
        self.hours = {}
        self.replay_bins = {}
        self.match_bins = {}
        self.current_loaded = False
        self.current_caught_up = False
        self.error = None
        self.events = []
        self.sequence = 0
        self.lanes = {}
        self.rate_events = []
        self.rate_lanes = {}
        self.rate_origin = small(self.campaign / "started.json").get("unix_seconds")

    def _old_lines(self):
        campaign = self.campaign
        seen = set()
        first = True
        while campaign:
            campaign = campaign.resolve()
            if campaign in seen:
                raise ValueError('campaign ancestry cycle')
            seen.add(campaign)
            options = small(campaign / 'config.json')
            plan = small(campaign / 'plan.json')
            current = first
            end = self.bootstrap_end if current else complete_end(campaign / 'training.log')
            first = False
            prior = small(Path(options['resume_progress'])) if options.get('resume_progress') else {}
            progress = (small(campaign/'training/durable-progress.json') or small(campaign/'training/progress.json')) if not current else {}
            skip_tail = not current and options.get('checkpoint_seconds',0)>0
            count = 0
            newest = True
            for line in reverse_lines(campaign / 'training.log', end):
                if skip_tail:
                    if b'"lane":' not in line or json.loads(line)['id'] != progress.get('last_game_id'):
                        continue
                    skip_tail = False
                if b'"lane":' in line:
                    count += 1
                    if newest:
                        newest = False
                        latest_id = json.loads(line)['id']
                        durable_id = progress.get('last_game_id')
                        if durable_id is not None and durable_id != latest_id:
                            # A controlled stop can land between durable receipt/
                            # progress publication and stdout. Recover that exact
                            # last receipt, without enumerating the game archive.
                            receipt = campaign/'training/games'/f'game-{durable_id:07}.json'
                            row = small(receipt)
                            latest = campaign/'training/games'/f'game-{latest_id:07}.json'
                            if row.get('id') != durable_id or receipt.stat().st_mtime < latest.stat().st_mtime:
                                raise ValueError('inconsistent archived final receipt')
                            yield campaign, options.get('seed'), plan.get('hashes', {}).get('bin/paisho-gen5'), json.dumps(row).encode()
                            count += 1
                yield campaign, options.get('seed'), plan.get('hashes', {}).get('bin/paisho-gen5'), line
            if current:self.current_loaded = True
            if progress.get('completed') is not None:
                expected = progress['completed']-prior.get('completed', 0)
                if count != expected:
                    raise ValueError(f'archived receipt count mismatch: {campaign.name}')
            campaign = Path(prior['previous_campaign']) if prior.get('previous_campaign') else None

    def _record(self, campaign, seed, binary_hash, line, cutoff):
        if b'"lane":' not in line:
            return None
        row = json.loads(line)
        if row.get('reanalysis') or row.get('lane') == 'Reanalysis':
            return None
        published = (campaign / 'training/games' / f"game-{row['id']:07}.json").stat().st_mtime
        lane, outcome = row['lane'], row['outcome']
        terminal = outcome in ('Win(Host)', 'Win(Guest)', 'Draw')
        wins = losses = unresolved = 0
        if lane != 'Selfplay' and outcome.startswith('Win('):
            seat = candidate_seat(row, seed, binary_hash)
            if seat is None:
                unresolved = 1
            elif outcome == f'Win({seat})':
                wins = 1
            else:
                losses = 1
        delta = (1, int(terminal), wins, losses, int(outcome == 'Draw'), unresolved,
                 row.get('continuation_decisions', row.get('decisions', 0)), row.get('fresh_used', 0), row.get('replay_used', 0))
        if campaign == self.campaign and lane == 'Historical':
            key = (int(published // 600) * 600, row.get('reference_budget'))
            totals = self.replay_bins.setdefault(key, [0] * len(FIELDS))
            for i, value in enumerate(delta):totals[i] += value
        if campaign == self.campaign:
            generation = (row.get('opponent') or 'Gen3.1') if lane == 'Historical' else None
            key = (int(published // 600) * 600, lane, generation, row.get('reference_identity'), row.get('reference_budget'))
            totals = self.match_bins.setdefault(key, [0] * len(FIELDS))
            for i, value in enumerate(delta):totals[i] += value
        bucket = self.hours.setdefault(int(published // 3600) * 3600, {})
        totals = bucket.setdefault(lane, [0] * len(FIELDS))
        for i, value in enumerate(delta):
            totals[i] += value
        if published > cutoff:
            totals = self.lanes.setdefault(lane, [0] * len(FIELDS))
            for i, value in enumerate(delta):
                totals[i] += value
            heapq.heappush(self.events, (published, self.sequence, lane, delta))
            self.sequence += 1
        rate_cutoff = max(cutoff + 3000, self.rate_origin or float('-inf'))
        if published > rate_cutoff:
            totals = self.rate_lanes.setdefault(lane, [0] * len(FIELDS))
            for i, value in enumerate(delta):
                totals[i] += value
            heapq.heappush(self.rate_events, (published, self.sequence, lane, delta))
            self.sequence += 1
        return published > cutoff

    def update(self, upto, seed, binary_hash, now=None, budget_seconds=.05, max_rows=4096):
        now = time.time() if now is None else now
        # A reader may be created before the launcher publishes started.json.
        if self.rate_origin is None:
            self.rate_origin = small(self.campaign / "started.json").get("unix_seconds")
        cutoff = now - 3600
        started = time.process_time()
        count = 0
        while self.events and self.events[0][0] <= cutoff and time.process_time() - started < budget_seconds:
            _, _, lane, delta = heapq.heappop(self.events)
            for i, value in enumerate(delta):
                self.lanes[lane][i] -= value
        rate_cutoff = max(now - 600, self.rate_origin or float('-inf'))
        while self.rate_events and self.rate_events[0][0] <= rate_cutoff and time.process_time() - started < budget_seconds:
            _, _, lane, delta = heapq.heappop(self.rate_events)
            for i, value in enumerate(delta):
                self.rate_lanes[lane][i] -= value
        try:
            # Live receipts have priority over historical backfill.
            if self.offset < upto:
                with (self.campaign / 'training.log').open('rb') as stream:
                    stream.seek(self.offset)
                    while self.offset < upto and count < max_rows and time.process_time() - started < budget_seconds:
                        line = stream.readline(min(131073, upto - self.offset))
                        if not line.endswith(b'\n'):
                            break
                        if len(line) > 131072:
                            raise ValueError('oversized game receipt')
                        self._record(self.campaign, seed, binary_hash, line, cutoff)
                        self.offset = stream.tell()
                        count += 1
            while self.loading and count < max_rows and time.process_time() - started < budget_seconds:
                item = next(self.bootstrap, None)
                if item is None:
                    self.bootstrap.close()
                    self.loading = False
                    self.window_loaded = True
                    break
                if self._record(*item, cutoff) is False:
                    # Serialized publication: everything still to backfill is older.
                    self.window_loaded = True
                count += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error = str(exc)
        expired = bool(self.events and self.events[0][0] <= cutoff)
        ready = self.window_loaded and self.offset == upto and self.error is None and not expired
        self.current_caught_up = self.current_loaded and self.offset == upto
        self.rate_ready = (self.current_caught_up if self.rate_origin else ready) and self.error is None and not (self.rate_events and self.rate_events[0][0] <= rate_cutoff)
        return {'ready': ready, 'seconds': 3600, 'from_unix_seconds': cutoff,
                'to_unix_seconds': now, 'timestamp_basis': 'receipt-file-publication',
                'error': self.error, 'lanes': {
                    lane: {**dict(zip(FIELDS, values)), 'ready': ready and values[5] == 0}
                    for lane, values in self.lanes.items() if values[0]}}

    def throughput(self, now=None):
        now = time.time() if now is None else now
        seconds = min(600, max(0, now - self.rate_origin)) if self.rate_origin else 600
        ready = getattr(self, 'rate_ready', False) and seconds > 0
        return {'ready': ready, 'seconds': seconds, 'maximum_seconds': 600,
                'timestamp_basis': 'receipt-file-publication',
                'terminal_per_second': {
                    lane: values[1] / seconds if ready else None
                    for lane, values in self.rate_lanes.items()},
                'terminal': {lane: values[1] for lane, values in self.rate_lanes.items()}}

    def replay_history(self, now=None):
        now = time.time() if now is None else now
        return {'ready': self.current_caught_up,
                'error': self.error if not self.current_loaded else None,
                'seconds': 600, 'match_bins': [
                    {'start': stamp, 'end': stamp+600, 'lane': lane, 'generation': generation,
                     'reference_identity': reference, 'budget': budget, 'partial': stamp+600>now,
                     **dict(zip(FIELDS, values)), 'seats_known': values[5]==0}
                    for (stamp,lane,generation,reference,budget),values in sorted(self.match_bins.items(),
                        key=lambda item:(item[0][0], item[0][1], item[0][2] or '', item[0][3] or '', item[0][4] or 0),reverse=True)],
                'bins': [
                    {'start': stamp, 'end': stamp+600, 'budget': budget,
                     'partial': stamp+600>now,
                     **dict(zip(FIELDS, values)), 'seats_known': values[5]==0}
                    for (stamp,budget),values in sorted(self.replay_bins.items(),
                        key=lambda item:(item[0][0], item[0][1] or 0),reverse=True)]}

    def history(self, now=None):
        now = time.time() if now is None else now
        ready = not self.loading and self.error is None
        return {'ready': ready, 'error': self.error, 'hours': [
            {'start': hour, 'end': hour + 3600, 'partial': hour + 3600 > now,
             'lanes': {lane: {**dict(zip(FIELDS, values)), 'ready': values[5] == 0}
                       for lane, values in lanes.items()}}
            for hour, lanes in sorted(self.hours.items(), reverse=True)]}
