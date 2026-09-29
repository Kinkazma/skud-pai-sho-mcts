#!/usr/bin/env python3
"""Run a benchmark with the local training campaign temporarily paused."""

import argparse
import json
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from paisho_control import ControlConfig, state_file_lock


def run_offline_paused(config, command):
    """Benchmark an explicitly offline controller with a persisted settled pause.

    Holding its state lock prevents a cooperating controller from changing that
    pause during the command. Missing/mismatched state never implies permission.
    """
    with state_file_lock(config):
        state = json.loads(config.state_path.read_text())
        if (state.get("job_fingerprint") != config.fingerprint
                or state.get("desired") != "paused"
                or state.get("observed") != "paused"
                or state.get("process_pid") is not None):
            raise ValueError("offline benchmark requires the matching persisted pause without a PID")
        print("training_paused_for_benchmark=true controller=offline", flush=True)
        return command()


def run_paused(request, command):
    initial = request("status")
    restore = initial["desired"] == "running" and not initial["completed"]
    pause_revision = None
    try:
        if restore:
            paused = request("pause")
            pause_revision = paused["revision"]
        deadline = time.monotonic() + 60
        while True:
            state = request("status")
            if (state["observed"] == "paused" or state["completed"]) and not state.get("examples_busy", False):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Training or background exports did not become idle")
            time.sleep(0.1)
        # Let already-submitted GPU commands drain before the benchmark warmup.
        time.sleep(1)
        print("training_paused_for_benchmark=true", flush=True)
        return command()
    finally:
        if restore:
            current = request("status")
            if (current["desired"] == "paused" and not current["completed"]
                    and current["pid"] == initial["pid"]
                    and (pause_revision is None or current["revision"] == pause_revision)):
                resumed = request("resume")
                print(f"training_after_benchmark={resumed['observed']}", flush=True)
            else:
                print("training_state_changed_during_benchmark=preserved", flush=True)


def run_gen3_paused(path, command):
    """Use Gen3's durable pause and preserve user actions during the benchmark."""
    from paisho_gen3_controls import Gen3Controls, read
    control=Gen3Controls(Path(path).resolve())
    initial=read(path);restore=initial.get('state')=='running'
    identity=initial.get('action_id')
    settled = ('paused', 'completed')
    if not restore and (initial.get('state') not in settled or initial.get('pid')):
        raise ValueError('Gen3 benchmark requires a running, settled paused or completed campaign')
    try:
        if restore:control.request('pause')
        deadline=time.monotonic()+120
        while read(path).get('state')=='pausing':
            if time.monotonic()>deadline:raise RuntimeError('Gen3 did not finish durable pause')
            time.sleep(.1)
        current=read(path)
        if current.get('state') not in settled or current.get('pid'):
            raise ValueError('Gen3 pause changed before benchmark')
        print('training_inactive_for_benchmark=true lineage=gen3 state='+current['state'],flush=True)
        return command()
    finally:
        current=read(path)
        if restore and current.get('state')=='paused' and current.get('action_id')==identity:
            print('training_after_benchmark='+control.request('resume')['state'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gen3", action="store_true", help="Config is a Gen3 lineage control state")
    parser.add_argument("--offline-paused", action="store_true",
                        help="Use the locked persisted pause when the controller service is stopped")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    if options.gen3:
        return run_gen3_paused(options.config, lambda: subprocess.call(command))
    config = ControlConfig.load(options.config)

    def request(action):
        req = urllib.request.Request(
            f"http://{config.host}:{config.port}/api/{action}",
            method="GET" if action == "status" else "POST",
            headers={"X-Paisho-Control": config.control_token},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    if options.offline_paused:
        return run_offline_paused(config, lambda: subprocess.call(command))
    return run_paused(request, lambda: subprocess.call(command))


if __name__ == "__main__":
    raise SystemExit(main())
