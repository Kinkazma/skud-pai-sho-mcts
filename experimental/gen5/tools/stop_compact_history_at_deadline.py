#!/usr/bin/env python3
"""Stop only the identified comparison companion at the current training deadline.

One local process-lifetime guard. It never restarts training or installs a task.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from paisho_compact_campaign import atomic_json
from paisho_compact_history import process_identity


def read(path):
    return json.loads(path.read_text())


def stop_expected(pid, identity):
    if process_identity(pid) != identity:
        return "already-exited-or-identity-changed"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-exited"
    return "stop-signal-sent"


def run(campaign):
    plan = read(campaign / "history-stop-plan.json")
    end = dt.datetime.fromisoformat(plan["until"])
    while True:
        remaining = (end - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 10))
    result = stop_expected(plan["pid"], plan["process_identity"])
    # The companion's SIGTERM handler reaps its one owned comparison child.
    for _ in range(15):
        if process_identity(plan["pid"]) != plan["process_identity"]:
            break
        time.sleep(1)
    atomic_json(campaign / "history-stop-receipt.json", {"result": result,
                "companion_exited": process_identity(plan["pid"]) != plan["process_identity"],
                "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reason": "user cancelled extension; release evaluation CPUs at original one-hour deadline"})


def arm(campaign):
    state = read(campaign / "status.json")
    duration = read(campaign / "campaign.json")["seconds"]
    end = dt.datetime.fromisoformat(state["started_at"]) + dt.timedelta(seconds=duration)
    pid = read(campaign / "history-launch.json")["pid"]
    identity = process_identity(pid)
    if not identity:
        raise ValueError("comparison companion is not running")
    plan = {"until": end.isoformat(), "pid": pid, "process_identity": identity,
            "training_is_not_restarted_or_signalled": True}
    with (campaign / "history-stop-plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
    with (campaign / "history-stop.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run",
                                  "--campaign", str(campaign)], start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    atomic_json(campaign / "history-stop-launch.json", {"pid": child.pid, "until": end.isoformat()})
    return {"guard_pid": child.pid, **plan}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("arm", "run"))
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "arm":
        print(json.dumps(arm(args.campaign.resolve())))
    else:
        run(args.campaign.resolve())
