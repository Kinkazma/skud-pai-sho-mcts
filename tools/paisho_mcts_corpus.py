#!/usr/bin/env python3
"""Generate a resumable corpus with shared GPU evaluation and hard game deadlines.

Build: python3 tools/paisho_mcts_corpus.py build
Run:   python3 tools/paisho_mcts_corpus.py generate OUTPUT --games 10000
Hardware benchmarks must use benchmark_with_training_paused.py.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/paisho-mcts-corpus'
SERVICE = ROOT / 'target/release/paisho-mcts-metal'
KERNEL = ROOT / 'apple/paisho-mcts-metal/evaluate.metal'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build():
    subprocess.run(['cargo', 'build', '--release', '-p', 'paisho-train', '--bin', 'paisho-mcts-corpus'], cwd=ROOT, check=True)
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    subprocess.run(['swiftc', '-sdk', sdk, '-O', str(KERNEL.with_name('main.swift')), '-o', str(SERVICE)],
                   env={**os.environ, 'SDKROOT': sdk}, check=True)


def load_receipts(output):
    rows = []
    for path in sorted((output / 'attempts').glob('*/receipt.json')):
        row = json.loads(path.read_text())
        if row['status'] == 'accepted':
            game = path.parent / 'game.psr'
            if digest(game) != row['psr_sha256']:
                raise ValueError(f'corrupt corpus game: {game}')
            result = json.loads((path.parent / 'result.json').read_text())
            if digest(path.parent / 'result.json') != row['result_sha256'] or not result['terminal']:
                raise ValueError(f'corrupt corpus result: {path.parent}')
        rows.append(row)
    return rows


def game_deadline(args):
    if args.deadline is not None:
        return args.deadline
    plan_path = args.output.resolve() / 'plan.json'
    if plan_path.exists():
        # A new default must not rewrite or prevent resuming an existing corpus.
        return json.loads(plan_path.read_text())['settings']['deadline_seconds']
    if args.backend == 'cpu' and args.simulations == 32:
        return 8
    return 20 if args.simulations == 8 else 50


def generate(args):
    output = args.output.resolve()
    deadline = game_deadline(args)
    if min(args.games, args.workers, args.simulations, args.batch, args.source_limit) < 1 or not 0 < deadline < float('inf'):
        raise ValueError('counts and deadline must be positive and finite')
    if args.batch > 32768 or not 0 <= args.wait_us <= 10000 or args.first_id < 0:
        raise ValueError('invalid batch, wait or first ID')
    maximum = args.max_attempts if args.max_attempts is not None else args.games * 10
    if maximum < args.games:
        raise ValueError('max-attempts must be at least games')
    requested = {'version': 1, 'backend': args.backend, 'simulations': args.simulations,
                 'games': args.games, 'workers': args.workers, 'batch': args.batch,
                 'wait_us': args.wait_us, 'deadline_seconds': deadline, 'max_attempts': maximum,
                 'first_id': args.first_id, 'source_limit': args.source_limit,
                 'seed_protocol': '70332026 xor ordinal*0x9e3779b97f4a7c15, StableRng.next_u64',
                 'rules': 'skud-pai-sho-2022-03-14-v2', 'cuts': 'all nonterminal Main boundaries including opening',
                 'search': 'MctsConfig defaults except simulations; one independent tree; exhaustive action ranking'}
    output.mkdir(parents=True, exist_ok=True)
    # OS lock releases on crash; excludes two producers writing the same archive.
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['settings'] != requested:
                raise ValueError('resume settings differ from immutable corpus plan')
            for name, expected in plan['files'].items():
                if digest(output / 'bin' / name) != expected:
                    raise ValueError(f'changed frozen executable/source: {name}')
        else:
            (output / 'bin').mkdir(exist_ok=True)
            paths = [BINARY, Path(__file__).resolve()]
            if args.backend == 'metal':
                paths += [SERVICE, KERNEL, KERNEL.with_name('main.swift')]
            for path in paths:
                shutil.copy2(path, output / 'bin' / path.name)
            plan = {'settings': requested, 'files': {p.name: digest(p) for p in paths}}
            write_json(plan_path, plan)
        (output / 'attempts').mkdir(exist_ok=True)
        rows = load_receipts(output)
        # Any directory without a receipt was interrupted. Keep it as evidence;
        # never silently count it as a completed game or reuse its ordinal.
        used = [int(p.name) for p in (output / 'attempts').iterdir() if p.is_dir() and p.name.isdigit()]
        ordinal = max(used, default=args.first_id-1) + 1
        accepted = sum(row['status'] == 'accepted' for row in rows)
        daemon = None
        temporary = tempfile.TemporaryDirectory(prefix='paisho-mcts-')
        socket = str(Path(temporary.name) / 'gpu.sock')
        log = (output / 'service.log').open('a')
        started = time.monotonic()
        stopping = threading.Event()
        active = set()
        active_lock = threading.Lock()
        def interrupt(signum, frame):
            stopping.set()
            with active_lock:
                for process in active:
                    process.kill()
            raise KeyboardInterrupt
        previous_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
        def publish_summary():
            summary = {'accepted': accepted, 'target': args.games, 'recorded_attempts': len(rows),
                       'timeouts': sum(r['status'] == 'timeout' for r in rows),
                       'unfinished': sum(r['status'] == 'unfinished' for r in rows),
                       'elapsed_this_run_seconds': time.monotonic()-started,
                       'occupied_worker_seconds': sum(r['elapsed_seconds'] for r in rows),
                       'complete': accepted >= args.games}
            write_json(output / 'summary.json', summary)
            return summary
        publish_summary()
        try:
            if args.backend == 'metal' and accepted < args.games:
                command = [str(output / 'bin' / BINARY.name), 'serve', str(output / 'bin' / SERVICE.name),
                           str(output / 'bin' / KERNEL.name), socket, str(args.batch), str(args.wait_us)]
                daemon = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                ready_deadline = time.monotonic() + 35
                while not Path(socket).exists():
                    if daemon.poll() is not None or time.monotonic() > ready_deadline:
                        raise RuntimeError('GPU service failed to start; see service.log')
                    time.sleep(0.05)

            def job(identity):
                directory = output / 'attempts' / f'{identity:08d}'
                directory.mkdir()
                destination = directory / 'produced'
                begin = time.monotonic()
                command = [str(output / 'bin' / BINARY.name), 'game', str(args.simulations), str(identity),
                           socket if args.backend == 'metal' else 'cpu', str(destination), str(args.source_limit)]
                with (directory / 'worker.log').open('w') as worker_log:
                    try:
                        with active_lock:
                            if stopping.is_set():
                                raise RuntimeError('corpus interrupted')
                            process = subprocess.Popen(command, stdout=worker_log, stderr=worker_log)
                            active.add(process)
                        try:
                            returncode = process.wait(timeout=deadline)
                        except BaseException:
                            process.kill()
                            process.wait()
                            raise
                        finally:
                            with active_lock:
                                active.discard(process)
                        status = 'error' if returncode else 'unfinished'
                        if not returncode:
                            report = json.loads((destination / 'result.json').read_text())
                            if report['terminal']:
                                status = 'accepted'
                        # Wall deadline covers output publication as well as search.
                        if time.monotonic() - begin > deadline:
                            status = 'timeout'
                    except subprocess.TimeoutExpired:
                        # The wait handler kills and reaps this game process; its CPU
                        # search stops here; the shared GPU daemon stays alive.
                        status = 'timeout'
                row = {'ordinal': identity, 'status': status, 'elapsed_seconds': time.monotonic()-begin}
                if status == 'accepted':
                    for name in ('game.psr', 'result.json'):
                        (destination / name).replace(directory / name)
                    row.update(psr_sha256=digest(directory / 'game.psr'), result_sha256=digest(directory / 'result.json'),
                               decisions=report['decisions'], cuts=len(report['cuts']))
                write_json(directory / 'receipt.json', row)
                return row

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = set()
                while accepted < args.games:
                    if daemon is not None and daemon.poll() is not None:
                        raise RuntimeError('GPU service stopped; partial corpus preserved')
                    while len(pending) < min(args.workers, args.games-accepted) and ordinal < args.first_id+maximum:
                        pending.add(pool.submit(job, ordinal))
                        ordinal += 1
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = future.result()
                        rows.append(row)
                        accepted += row['status'] == 'accepted'
                        if row['status'] == 'error':
                            raise RuntimeError(f'game {row["ordinal"]} failed; see worker.log; corpus preserved')
                    summary = publish_summary()
                    print(json.dumps(summary), flush=True)
            if accepted < args.games:
                raise RuntimeError('attempt budget exhausted before corpus target; partial corpus preserved')
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
            if daemon is not None:
                try:
                    os.killpg(daemon.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                daemon.wait(timeout=10)
            metrics = Path(socket + '.metrics.json')
            if metrics.exists():
                shutil.copy2(metrics, output / 'gpu-metrics.json')
            log.close()
            temporary.cleanup()
            rows = load_receipts(output)
            accepted = sum(row['status'] == 'accepted' for row in rows)
            publish_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build')
    sample = commands.add_parser('sample')
    sample.add_argument('output', type=Path)
    sample.add_argument('destination', type=Path)
    sample.add_argument('--seed', type=int, required=True)
    run = commands.add_parser('generate')
    run.add_argument('output', type=Path)
    run.add_argument('--backend', choices=['cpu', 'metal'], default='metal')
    run.add_argument('--simulations', type=int, choices=[8, 32, 128, 512], default=32)
    run.add_argument('--games', type=int, default=10000)
    run.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    run.add_argument('--batch', type=int, default=4096)
    run.add_argument('--wait-us', type=int, default=0)
    run.add_argument('--deadline', type=float)
    run.add_argument('--max-attempts', type=int)
    run.add_argument('--first-id', type=int, default=20000)
    run.add_argument('--source-limit', type=int, default=16384)
    args = parser.parse_args()
    if args.command == 'build':
        build()
    elif args.command == 'sample':
        import random
        rows = load_receipts(args.output)
        rows = [r for r in rows if r['status'] == 'accepted']
        if not rows:
            raise ValueError('no completed games in corpus')
        rng = random.Random(args.seed)
        row = rng.choice(rows)
        directory = args.output / 'attempts' / f'{row["ordinal"]:08d}'
        report = json.loads((directory / 'result.json').read_text())
        cut = rng.choice(report['cuts'])
        subprocess.run([str(args.output.resolve() / 'bin' / BINARY.name), 'cut',
                        str(directory / 'game.psr'), str(cut), str(args.destination)], check=True)
        print(json.dumps({'source_ordinal': row['ordinal'], 'cut_decisions': cut,
                          'remaining_decisions': report['decisions']-cut, 'source_sha256': row['psr_sha256']}))
    else:
        generate(args)


if __name__ == '__main__':
    main()
