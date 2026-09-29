"""Model-relative results from the existing append-only Gen5 receipt journal.

The first frozen runtime omitted candidate_seat. Its documented SplitMix64 seat
schedule is recoverable exactly from seed + game ID. Pin that compatibility path
to the executable hash; newer receipts must supply their explicit seat.
"""
import json
from collections import deque
from pathlib import Path

LEGACY_SEAT_BINARY = 'a4f1b39e2c37bb39c4a81b6e2009ea742669be776842e3dae23be760b6be3e10'
MASK = (1 << 64) - 1


def legacy_seat(seed, game_id):
    state = (seed + game_id) & MASK
    def index(bound):
        nonlocal state
        rejection = (1 << 64) % bound
        while True:
            state = (state + 0x9e3779b97f4a7c15) & MASK
            value = state
            value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
            value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
            value ^= value >> 31
            if value >= rejection:
                return value % bound
    index(114)  # Standard setup draw, before the seat draw in collector::play.
    return 'Host' if index(2) == 0 else 'Guest'


def candidate_seat(row, seed, binary_hash):
    seat = row.get('candidate_seat')
    if seat in ('Host', 'Guest'):
        return seat
    if seat is None and binary_hash == LEGACY_SEAT_BINARY and type(seed) is int:
        return legacy_seat(seed, row['id'])
    return None


def describe_game(row, seed, binary_hash):
    same = row.get('lane') == 'Selfplay'
    seat = candidate_seat(row, seed, binary_hash)
    opponent = row.get('opponent') or ('Gen3.1' if row.get('lane') == 'Historical' else 'Version retenue')
    outcome = row.get('outcome')
    reason = {'wall-limit':'temps','decision-limit':'limite de coups','repetition-training-loss':'boucle','engine-error':'erreur'}.get(row.get('termination'))
    interrupted = 'Interrompue'+(f' ({reason})' if reason else '')
    if same:
        result = 'Terminée' if outcome in ('Win(Host)', 'Win(Guest)', 'Draw') else interrupted
    elif outcome in ('Win(Host)', 'Win(Guest)'):
        winner = outcome[4:-1]
        model = 'Gen5' if same or winner == seat else opponent if seat else 'Place'
        result = f'{model} gagne' if seat else f'Place {winner} gagne (modèle à identifier)'
    else:
        result = 'Partie nulle' if outcome == 'Draw' else interrupted
    return {'gen5_seat': 'Les deux' if same else seat, 'result_label': result}


class SeatResults:
    """Tail at most four MiB per refresh; never enumerate game/model directories."""
    def __init__(self, log):
        self.log = Path(log)
        self.offset = 0
        self.processed = 0
        self.lanes = {}
        self.recent = {}
        self.last_decisive = {}

    def update(self, completed, seed, binary_hash):
        if completed < self.processed:
            raise ValueError('campaign progress moved backwards')
        if not self.log.exists():
            return
        with self.log.open('rb') as stream:
            if stream.seek(0, 2) < self.offset:
                raise ValueError('campaign journal was truncated')
            stream.seek(self.offset)
            start = self.offset
            while self.processed < completed and self.offset - start < 4 * 1024 * 1024:
                line = stream.readline(131073)
                if not line or not line.endswith(b'\n'):
                    break  # Writer may still be appending the final line.
                if len(line) > 131072:
                    raise ValueError('oversized game receipt')
                self.offset = stream.tell()
                if b'"lane":' not in line:
                    continue
                self.processed += 1
                if b'"lane":"Selfplay"' in line:
                    continue
                row = json.loads(line)
                if row['lane'] in ('Selfplay', 'Reanalysis'):
                    continue
                self.recent.setdefault(row['lane'], deque(maxlen=3)).append(row)
                totals = self.lanes.setdefault(row['lane'], {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'unresolved_seat': 0})
                totals['games'] += 1
                outcome = row['outcome']
                if outcome == 'Draw':
                    totals['draws'] += 1
                elif outcome in ('Win(Host)', 'Win(Guest)'):
                    self.last_decisive[row['lane']] = row
                    seat = candidate_seat(row, seed, binary_hash)
                    if seat is None:
                        totals['unresolved_seat'] += 1
                    else:
                        totals['wins' if outcome == f'Win({seat})' else 'losses'] += 1

    def snapshot(self, counters):
        result = {}
        for lane, counter in counters.items():
            if lane in ('Selfplay', 'Reanalysis'):
                continue
            totals = self.lanes.get(lane, {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'unresolved_seat': 0})
            ready = totals['games'] == counter['games'] and totals['unresolved_seat'] == 0
            result[lane] = {**totals, 'ready': ready}
        return result
