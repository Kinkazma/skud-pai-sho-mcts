#!/usr/bin/env python3
"""Local, restartable control panel for one durable Pai Sho training command."""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import plistlib
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

if __package__:
    from .paisho_live_control import read_live_layout
    from .paisho_game_gallery import ArchiveGallery, Gallery, backfill as backfill_examples
else:
    from paisho_live_control import read_live_layout
    from paisho_game_gallery import ArchiveGallery, Gallery, backfill as backfill_examples


SCHEMA_VERSION = 1
DESIRED_STATES = {"paused", "running"}
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-g\d+-s(\d+)-a\d+\.psckpt$")
CHECKPOINT_ID_PATTERN = re.compile(
    r"^checkpoint-g(\d+)-s(\d+)-a\d+\.psckpt$"
)
GENERATION_DIRECTORY_PATTERN = re.compile(r"^generation-\d+$")
_STATE_LOCKS: dict[Path, threading.RLock] = {}
_STATE_LOCKS_GUARD = threading.Lock()


class ControlError(RuntimeError):
    pass


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ControlError(f"{name} must be a non-empty string")
    return value


def _optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    return Path(_require_string(value, name)).expanduser().resolve()


@dataclass(frozen=True)
class DashboardConfig:
    html_path: Path
    refresh_command: tuple[str, ...] | None
    refresh_working_directory: Path | None
    minimum_refresh_seconds: float
    curriculum_directory: Path | None
    internal_ratings_path: Path | None
    examples_archive_root: Path | None = None
    examples_directory: Path | None = None
    examples_campaign_directory: Path | None = None
    examples_exporter: Path | None = None


@dataclass(frozen=True)
class ControlConfig:
    path: Path
    name: str
    command: tuple[str, ...]
    working_directory: Path
    state_path: Path
    log_path: Path
    progress_directory: Path | None
    target_training_step: int | None
    host: str
    port: int
    control_token: str
    restart_delay_seconds: float
    dashboard: DashboardConfig | None
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "ControlConfig":
        path = path.expanduser().resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlError(f"cannot read control configuration {path}: {error}") from error
        if raw.get("version") != SCHEMA_VERSION:
            raise ControlError(f"unsupported configuration version in {path}")
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ControlError("command must be a non-empty array of strings")
        working_directory = Path(
            _require_string(raw.get("working_directory"), "working_directory")
        ).expanduser().resolve()
        state_path = Path(
            _require_string(raw.get("state_path"), "state_path")
        ).expanduser().resolve()
        log_path = Path(_require_string(raw.get("log_path"), "log_path")).expanduser().resolve()
        progress_directory = _optional_path(raw.get("progress_directory"), "progress_directory")
        target = raw.get("target_training_step")
        if target is not None and (not isinstance(target, int) or isinstance(target, bool) or target <= 0):
            raise ControlError("target_training_step must be a positive integer")
        host = raw.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            raise ControlError("the control server must bind to 127.0.0.1")
        port = raw.get("port", 8765)
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ControlError("port must be between 1 and 65535")
        token = _require_string(raw.get("control_token"), "control_token")
        restart_delay = raw.get("restart_delay_seconds", 5.0)
        if not isinstance(restart_delay, (int, float)) or restart_delay < 0:
            raise ControlError("restart_delay_seconds must be non-negative")
        dashboard = _parse_dashboard(raw.get("dashboard"))
        fingerprint_source = json.dumps(
            {
                "command": command,
                "working_directory": str(working_directory),
                "progress_directory": str(progress_directory) if progress_directory else None,
                "target_training_step": target,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        import hashlib

        return cls(
            path=path,
            name=_require_string(raw.get("name"), "name"),
            command=tuple(command),
            working_directory=working_directory,
            state_path=state_path,
            log_path=log_path,
            progress_directory=progress_directory,
            target_training_step=target,
            host=host,
            port=port,
            control_token=token,
            restart_delay_seconds=float(restart_delay),
            dashboard=dashboard,
            fingerprint=hashlib.sha256(fingerprint_source).hexdigest(),
        )


def _parse_dashboard(raw: Any) -> DashboardConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ControlError("dashboard must be an object")
    command = raw.get("refresh_command")
    if command is not None and (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ControlError("dashboard.refresh_command must be an array of strings")
    interval = raw.get("minimum_refresh_seconds", 5.0)
    if not isinstance(interval, (int, float)) or interval <= 0:
        raise ControlError("dashboard.minimum_refresh_seconds must be positive")
    return DashboardConfig(
        html_path=Path(
            _require_string(raw.get("html_path"), "dashboard.html_path")
        ).expanduser().resolve(),
        refresh_command=tuple(command) if command else None,
        refresh_working_directory=_optional_path(
            raw.get("refresh_working_directory"),
            "dashboard.refresh_working_directory",
        ),
        minimum_refresh_seconds=float(interval),
        curriculum_directory=_optional_path(
            raw.get("curriculum_directory"),
            "dashboard.curriculum_directory",
        ),
        internal_ratings_path=_optional_path(
            raw.get("internal_ratings_path"),
            "dashboard.internal_ratings_path",
        ),
        examples_archive_root=_optional_path(raw.get("examples_archive_root"), "dashboard.examples_archive_root"),
        examples_directory=_optional_path(
            raw.get("examples_directory"), "dashboard.examples_directory",
        ),
        examples_campaign_directory=_optional_path(
            raw.get("examples_campaign_directory"), "dashboard.examples_campaign_directory",
        ),
        examples_exporter=_optional_path(raw.get("examples_exporter"), "dashboard.examples_exporter"),
    )


def default_state(config: ControlConfig) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "job_fingerprint": config.fingerprint,
        "desired": "paused",
        "observed": "paused",
        "completed": False,
        "revision": 0,
        "updated_unix_seconds": int(time.time()),
        "process_pid": None,
        "process_started_unix_seconds": None,
        "process_start_identity": None,
        "last_exit_code": None,
        "last_error": None,
    }


def load_state(config: ControlConfig) -> dict[str, Any]:
    try:
        raw = json.loads(config.state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_state(config)
    except (OSError, json.JSONDecodeError) as error:
        raise ControlError(f"cannot read state {config.state_path}: {error}") from error
    if raw.get("version") != SCHEMA_VERSION:
        raise ControlError(f"unsupported state version in {config.state_path}")
    if raw.get("job_fingerprint") != config.fingerprint:
        return default_state(config)
    if raw.get("desired") not in DESIRED_STATES:
        raise ControlError(f"invalid desired state in {config.state_path}")
    return {**default_state(config), **raw}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{secrets.token_hex(4)}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def mutate_state(
    config: ControlConfig,
    mutation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    with state_file_lock(config):
        state = load_state(config)
        mutation(state)
        _publish_locked_state(config, state)
        return state


@contextlib.contextmanager
def state_file_lock(config: ControlConfig):
    import fcntl

    with _STATE_LOCKS_GUARD:
        thread_lock = _STATE_LOCKS.setdefault(config.state_path, threading.RLock())
    lock_path = config.state_path.with_suffix(config.state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with thread_lock, lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _publish_locked_state(config: ControlConfig, state: dict[str, Any]) -> None:
    state["version"] = SCHEMA_VERSION
    state["job_fingerprint"] = config.fingerprint
    state["revision"] = int(state.get("revision", 0)) + 1
    state["updated_unix_seconds"] = int(time.time())
    atomic_write_json(config.state_path, state)


@dataclass(frozen=True)
class Progress:
    completed_step: int
    target_step: int | None
    checkpoint_count: int
    latest_checkpoint: str | None
    live: dict[str, Any] | None = None

    @property
    def target_reached(self) -> bool:
        return (
            self.target_step is not None
            and self.completed_step >= self.target_step
            and not (self.live and self.live.get("assessment_pending"))
        )

    @property
    def percent(self) -> float | None:
        if self.target_step is None:
            return None
        return min(100.0, 100.0 * self.completed_step / self.target_step)


def read_progress(config: ControlConfig) -> Progress:
    target = config.target_training_step
    directory = config.progress_directory
    if directory is None or not directory.is_dir():
        return Progress(0, target, 0, None)
    live = read_live_layout(directory)
    if live is not None:
        # Live completion is the command's successful exit, not a monotone Adam step:
        # promotion can restore an older champion's optimizer state.
        return Progress(live["completed_step"], None, live["checkpoint_count"], live["latest_checkpoint"], live)
    checkpoints: list[tuple[int, Path]] = []
    for learner_directory in _learner_progress_directories(directory):
        committed_steps: set[int] = set()
        for commit in learner_directory.glob("commit-s*.pslearn"):
            match = re.fullmatch(r"commit-s(\d+)\.pslearn", commit.name)
            if match:
                committed_steps.add(int(match.group(1)))
        for path in learner_directory.glob("checkpoint-g*-s*-a*.psckpt"):
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match:
                step = int(match.group(1))
                if step in committed_steps:
                    checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        return Progress(0, target, 0, None)
    step, latest = checkpoints[-1]
    return Progress(step, target, len(checkpoints), str(latest))


def _learner_progress_directories(directory: Path) -> list[Path]:
    learners = [directory]
    generations = sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and GENERATION_DIRECTORY_PATTERN.fullmatch(child.name)
    )
    learners.extend(
        learner
        for generation in generations
        if (learner := generation / "learner").is_dir()
    )
    return learners


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _process_start_identity(pid: int) -> str | None:
    if not _pid_is_alive(pid):
        return None
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    identity = " ".join(result.stdout.split())
    return identity or None


def _pid_matches_identity(pid: int, expected_identity: Any) -> bool:
    return isinstance(expected_identity, str) and (
        _process_start_identity(pid) == expected_identity
    )


def _signal_group(pid: int, requested_signal: signal.Signals) -> None:
    try:
        os.killpg(pid, requested_signal)
    except ProcessLookupError:
        return


def _process_usage(pid: int | None) -> dict[str, Any]:
    if pid is None or not _pid_is_alive(pid):
        return {"cpu_percent": 0.0, "memory_percent": 0.0, "processes": 0}
    result = subprocess.run(
        ["/bin/ps", "-Ao", "pgid=,%cpu=,%mem="],
        check=False,
        capture_output=True,
        text=True,
    )
    cpu = 0.0
    memory = 0.0
    count = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pgid, item_cpu, item_memory = int(fields[0]), float(fields[1]), float(fields[2])
        except ValueError:
            continue
        if pgid == pid:
            cpu += item_cpu
            memory += item_memory
            count += 1
    return {
        "cpu_percent": round(cpu, 1),
        "memory_percent": round(memory, 1),
        "processes": count,
    }


def _tail(path: Path, maximum_bytes: int = 24_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - maximum_bytes))
            data = stream.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-maximum_bytes:]


class LiveCampaignMetrics:
    """Cheap, cached status derived from small immutable summary files."""

    def __init__(self, dashboard: DashboardConfig | None):
        self.dashboard = dashboard
        self._decision_signature: tuple[int, str] | None = None
        self._decision_summary: dict[str, Any] = {}
        self._ratings_signature: tuple[int, int] | None = None
        self._ratings: list[dict[str, Any]] = []

    def read(self, progress: Progress) -> dict[str, Any]:
        if progress.live is not None:
            return dict(progress.live)
        active_generation = None
        if progress.latest_checkpoint:
            match = CHECKPOINT_ID_PATTERN.fullmatch(Path(progress.latest_checkpoint).name)
            if match:
                active_generation = int(match.group(1))
        decisions = self._read_decisions()
        ratings = self._read_ratings()
        return {
            "active_generation": active_generation,
            "latest_evaluation": decisions.get("latest"),
            "best_comparable_evaluation": decisions.get("best"),
            "internal_rating": None,
            "internal_rating_status": "bridge-pending",
            "rating_references": ratings,
        }

    def _read_decisions(self) -> dict[str, Any]:
        directory = (
            self.dashboard.curriculum_directory / "decisions"
            if self.dashboard and self.dashboard.curriculum_directory
            else None
        )
        if directory is None or not directory.is_dir():
            return {}
        paths = sorted(directory.glob("decision-*.json"))
        signature = (len(paths), paths[-1].name if paths else "")
        if signature == self._decision_signature:
            return self._decision_summary
        observations = []
        for path in paths:
            observation = self._read_decision(path)
            if observation is not None:
                observations.append(observation)
        latest = max(observations, key=lambda item: item["generation"], default=None)
        best = None
        if latest is not None:
            comparable = [
                item
                for item in observations
                if item["opponent"] == latest["opponent"]
                and item["elo_gap"] is not None
            ]
            best = max(comparable, key=lambda item: item["elo_gap"], default=None)
        self._decision_signature = signature
        self._decision_summary = {"latest": latest, "best": best}
        return self._decision_summary

    @staticmethod
    def _read_decision(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            evidence = payload["evidence"]
            if evidence.get("kind") != "fixed-opponent":
                return None
            point = evidence.get("contextual_elo_point", {})
            elo_gap = point.get("elo") if point.get("kind") == "finite" else None
            davidson = evidence.get("davidson_candidate_minus_opponent") or {}
            interval = davidson.get("paired_cluster") or {}
            checkpoint = payload.get("checkpoint") or {}
            return {
                "generation": int(payload["generation"]),
                "opponent": str(payload["tier_before"]),
                "elo_gap": float(elo_gap) if elo_gap is not None else None,
                "davidson_gap": _optional_float(davidson.get("estimate")),
                "ci95_low": _optional_float(interval.get("interval_95_lower")),
                "ci95_high": _optional_float(interval.get("interval_95_upper")),
                "eligible_pairs": int(evidence.get("eligible_pairs", 0)),
                "conclusion": str(evidence.get("conclusion", "unknown")),
                "checkpoint_sha256": checkpoint.get("sha256"),
            }
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None

    def _read_ratings(self) -> list[dict[str, Any]]:
        path = self.dashboard.internal_ratings_path if self.dashboard else None
        if path is None:
            return []
        try:
            stat = path.stat()
        except OSError:
            return []
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._ratings_signature:
            return self._ratings
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            header_index = next(
                index for index, line in enumerate(lines) if line.startswith("alias\tagent_id\t")
            )
            header = lines[header_index].split("\t")
            rows = []
            for line in lines[header_index + 1 :]:
                fields = line.split("\t")
                if len(fields) != len(header):
                    continue
                item = dict(zip(header, fields))
                rows.append(
                    {
                        "alias": item["alias"],
                        "elo": float(item["elo"]),
                        "ci95_low": float(item["paired_cluster_ci95_low"]),
                        "ci95_high": float(item["paired_cluster_ci95_high"]),
                        "status": item["rating_status"],
                    }
                )
        except (OSError, StopIteration, KeyError, ValueError):
            return self._ratings
        self._ratings_signature = signature
        self._ratings = rows
        return self._ratings


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


class Supervisor:
    def __init__(self, config: ControlConfig):
        self.config = config
        self.child: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._next_start_at = 0.0
        self._live_metrics = LiveCampaignMetrics(config.dashboard)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.reconcile()
            except Exception as error:  # keep the local controller available
                self._record_error(f"controller reconciliation failed: {error}")

    def reconcile(self) -> None:
        with self._lock:
            if self.child is not None:
                exit_code = self.child.poll()
                if exit_code is not None:
                    exited_pid = self.child.pid
                    self.child = None
                    self._record_exit(exited_pid, exit_code)
                    return
            state = load_state(self.config)
            pid = self._active_pid(state)
            progress = read_progress(self.config)
            if (
                pid is None
                and
                progress.target_reached
            ):
                self._mark_completed_if_needed(state)
                return
            if pid is not None:
                if state["desired"] == "paused" and state["observed"] != "paused":
                    _signal_group(pid, signal.SIGSTOP)
                    self._update_observed("paused", pid=pid)
                elif state["desired"] == "running" and state["observed"] != "running":
                    _signal_group(pid, signal.SIGCONT)
                    self._update_observed("running", pid=pid)
                return
            if state["process_pid"] is not None:
                self._record_exit(state["process_pid"], None)
                state = load_state(self.config)
            if state["desired"] == "running" and time.monotonic() >= self._next_start_at:
                self._start_child_if_requested()
            elif state["desired"] == "paused" and state["observed"] != "paused":
                self._update_observed("paused", pid=None)

    def _active_pid(self, state: dict[str, Any]) -> int | None:
        if self.child is not None:
            if self.child.poll() is None:
                return self.child.pid
            return None
        pid = state.get("process_pid")
        if isinstance(pid, int) and _pid_matches_identity(
            pid,
            state.get("process_start_identity"),
        ):
            return pid
        return None

    def _start_child_if_requested(self) -> None:
        started_pid = None
        with state_file_lock(self.config):
            state = load_state(self.config)
            progress = read_progress(self.config)
            if state["desired"] != "running":
                return
            if (
                progress.target_reached
            ):
                state.update(
                    desired="paused",
                    observed="completed",
                    completed=True,
                    process_pid=None,
                    process_started_unix_seconds=None,
                    process_start_identity=None,
                    last_exit_code=0,
                    last_error=None,
                )
                _publish_locked_state(self.config, state)
                return
            existing_pid = state.get("process_pid")
            if isinstance(existing_pid, int) and _pid_matches_identity(
                existing_pid,
                state.get("process_start_identity"),
            ):
                return
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.config.log_path.open("ab", buffering=0)
            try:
                child = subprocess.Popen(
                    self.config.command,
                    cwd=self.config.working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log.close()
            start_identity = _process_start_identity(child.pid)
            if start_identity is None:
                _signal_group(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                raise ControlError("cannot identify the newly started training process")
            self.child = child
            started_pid = child.pid
            state.update(
                observed="running",
                completed=False,
                process_pid=child.pid,
                process_started_unix_seconds=int(time.time()),
                process_start_identity=start_identity,
                last_exit_code=None,
                last_error=None,
            )
            _publish_locked_state(self.config, state)
        if started_pid is not None:
            print(f"training_started pid={started_pid} name={self.config.name}", flush=True)

    def _record_exit(self, pid: int, exit_code: int | None) -> None:
        successful = exit_code == 0

        def update(state: dict[str, Any]) -> None:
            if state.get("process_pid") != pid:
                return
            state["process_pid"] = None
            state["process_started_unix_seconds"] = None
            state["process_start_identity"] = None
            state["last_exit_code"] = exit_code
            if successful:
                state["desired"] = "paused"
                state["observed"] = "completed"
                state["completed"] = True
                state["last_error"] = None
            elif state["desired"] == "running":
                state["observed"] = "retrying"
                state["completed"] = False
                state["last_error"] = (
                    f"training exited with code {exit_code}"
                    if exit_code is not None
                    else "training process disappeared; restart scheduled"
                )
            else:
                state["observed"] = "paused"

        mutate_state(self.config, update)
        if not successful:
            self._next_start_at = time.monotonic() + self.config.restart_delay_seconds
        print(f"training_exited code={exit_code}", flush=True)

    def _mark_completed_if_needed(self, state: dict[str, Any]) -> None:
        pid = self._active_pid(state)
        if pid is not None:
            return
        if state["completed"] and state["observed"] == "completed":
            return

        def update(current: dict[str, Any]) -> None:
            current.update(
                desired="paused",
                observed="completed",
                completed=True,
                process_pid=None,
                process_started_unix_seconds=None,
                process_start_identity=None,
                last_exit_code=0,
                last_error=None,
            )

        mutate_state(self.config, update)

    def _update_observed(self, observed: str, pid: int | None) -> None:
        def update(state: dict[str, Any]) -> None:
            state["observed"] = observed
            state["process_pid"] = pid
            state["last_error"] = None

        mutate_state(self.config, update)

    def _record_error(self, message: str) -> None:
        def update(state: dict[str, Any]) -> None:
            state["last_error"] = message
            if state["desired"] == "running":
                state["observed"] = "retrying"

        try:
            mutate_state(self.config, update)
        except Exception:
            print(message, file=sys.stderr, flush=True)

    def set_desired(self, desired: str) -> dict[str, Any]:
        if desired not in DESIRED_STATES:
            raise ControlError(f"invalid desired state {desired}")

        def update(state: dict[str, Any]) -> None:
            state["desired"] = desired
            state["completed"] = False if desired == "running" else state["completed"]
            state["last_error"] = None

        with self._lock:
            mutate_state(self.config, update)
            self.reconcile()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = load_state(self.config)
            progress = read_progress(self.config)
            pid = self._active_pid(state)
            usage = _process_usage(pid)
            return {
                "name": self.config.name,
                "desired": state["desired"],
                "observed": state["observed"],
                "completed": state["completed"],
                "revision": state["revision"],
                "updated_unix_seconds": state["updated_unix_seconds"],
                "pid": pid,
                "process_started_unix_seconds": state["process_started_unix_seconds"],
                "last_exit_code": state["last_exit_code"],
                "last_error": state["last_error"],
                "progress": {
                    "completed_step": progress.completed_step,
                    "target_step": progress.target_step,
                    "percent": progress.percent,
                    "checkpoint_count": progress.checkpoint_count,
                    "latest_checkpoint": progress.latest_checkpoint,
                },
                "usage": usage,
                "campaign_metrics": self._live_metrics.read(progress),
                "log_tail": _tail(self.config.log_path),
                "dashboard_mtime_ns": _mtime_ns(
                    self.config.dashboard.html_path if self.config.dashboard else None
                ),
            }


def _mtime_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


class DashboardProvider:
    def __init__(self, config: DashboardConfig | None, examples_allowed: Callable[[], bool] = lambda: True):
        self.config = config
        self._examples_allowed = examples_allowed
        self._lock = threading.Lock()
        self._last_refresh = time.monotonic()
        self._refresh_in_progress = False
        self._last_examples_refresh = 0.0
        self._examples_in_progress = False

    def read(self) -> bytes:
        if self.config is None:
            return _missing_dashboard("Aucun tableau de campagne n'est configuré.")
        self.request_refresh()
        try:
            page = self.config.html_path.read_text(encoding="utf-8")
        except OSError as error:
            return _missing_dashboard(f"Tableau indisponible : {error}")
        refresh = '<meta http-equiv="refresh" content="10">'
        if "http-equiv=\"refresh\"" in page:
            page = re.sub(
                r'<meta http-equiv="refresh" content="\d+">',
                refresh,
                page,
                count=1,
            )
        else:
            page = page.replace("</head>", f"{refresh}</head>", 1)
        return page.encode("utf-8")

    def request_refresh(self) -> None:
        if self.config is None:
            return
        with self._lock:
            now = time.monotonic()
            if (self.config.examples_directory and self.config.examples_campaign_directory
                    and not self._examples_in_progress
                    and now - self._last_examples_refresh >= 60 and self._examples_allowed()):
                self._examples_in_progress = True
                threading.Thread(target=self._refresh_examples, name="campaign-psr-export",
                                 daemon=True).start()
            if (
                self.config.refresh_command
                and not self._refresh_in_progress
                and now - self._last_refresh >= self.config.minimum_refresh_seconds
            ):
                self._refresh_in_progress = True
                threading.Thread(
                    target=self._refresh,
                    name="campaign-dashboard-refresh",
                    daemon=True,
                ).start()
    def _refresh(self) -> None:
        assert self.config is not None
        try:
            if self.config.refresh_command:
                subprocess.run(
                    self.config.refresh_command,
                    cwd=self.config.refresh_working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                    check=True,
                )
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
            pass
        finally:
            with self._lock:
                self._last_refresh = time.monotonic()
                self._refresh_in_progress = False

    def _refresh_examples(self) -> None:
        assert self.config is not None
        try:
            if self.config.examples_directory and self.config.examples_campaign_directory:
                backfill_examples(
                    self.config.examples_campaign_directory, self.config.examples_directory,
                    self.config.examples_exporter or Path(__file__).resolve().parents[1] / "target/release/paisho-export-games",
                    jobs=1, allowed=self._examples_allowed,
                )
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"game_examples_warning={error}", flush=True)
        finally:
            with self._lock:
                self._last_examples_refresh = time.monotonic()
                self._examples_in_progress = False

    @property
    def examples_busy(self) -> bool:
        with self._lock:
            return self._examples_in_progress


def _missing_dashboard(message: str) -> bytes:
    return (
        "<!doctype html><html lang=\"fr\"><meta charset=\"utf-8\">"
        f"<body><p>{html.escape(message)}</p></body></html>"
    ).encode("utf-8")


def render_control_page(config: ControlConfig) -> bytes:
    name = html.escape(config.name)
    token = json.dumps(config.control_token)
    examples_link = '<p><a href="/examples" style="color:var(--jade)">Parties par génération · Télécharger / copier PSR</a></p>' if config.dashboard and (config.dashboard.examples_directory or config.dashboard.examples_archive_root) else ""
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pai Sho · Contrôle d'entraînement</title>
<style>
:root{{color-scheme:dark;--bg:#0d1510;--panel:#17231b;--line:#34483a;--ink:#f2efdf;--muted:#aebbac;--jade:#79c995;--gold:#e7bd67;--red:#ef8d75}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{width:min(1180px,calc(100% - 28px));margin:28px auto 60px}}header,.controls,.metrics{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
header{{justify-content:space-between;margin-bottom:16px}}h1{{font-family:Georgia,serif;font-size:clamp(1.7rem,4vw,3rem);margin:4px 0}}.eyebrow{{color:var(--gold);font-size:.72rem;letter-spacing:.16em;font-weight:800;margin:0}}
.panel,.metric{{background:linear-gradient(145deg,#1e3024,#142018);border:1px solid var(--line);border-radius:18px}}.panel{{padding:18px;margin:12px 0}}.metrics{{align-items:stretch}}.metric{{padding:15px;min-width:150px;flex:1}}.metric small{{color:var(--muted);display:block}}.metric strong{{font:1.55rem Georgia,serif;display:block;margin-top:8px}}
button{{border:1px solid var(--line);border-radius:12px;padding:11px 16px;color:var(--ink);font-weight:750;background:#24382a;cursor:pointer}}button.primary{{background:var(--jade);color:#0d1510}}button.danger{{border-color:#824f43;color:#ffd3c8}}button:disabled{{opacity:.4;cursor:default}}
.state{{font-weight:800;color:var(--jade)}}.state.paused{{color:var(--gold)}}.state.retrying,.state.failed{{color:var(--red)}}progress{{width:100%;height:14px;accent-color:var(--jade)}}
.note{{color:var(--muted);font-size:.82rem;line-height:1.5}}.action-status{{min-height:1.5em;color:var(--gold)}}.archive-head{{margin:26px 2px 10px}}.archive-head h2{{font:1.6rem Georgia,serif;margin:4px 0}}pre{{max-height:220px;overflow:auto;background:#080d09;border:1px solid var(--line);border-radius:12px;padding:12px;color:#cbd9cd;font-size:.72rem;white-space:pre-wrap}}
iframe{{width:100%;height:900px;border:1px solid var(--line);border-radius:18px;background:#101812}}@media(max-width:650px){{iframe{{height:720px}}}}
</style></head><body><main>
<header><div><p class="eyebrow">PAI SHO · CONTRÔLE LOCAL</p><h1>{name}</h1></div><div class="controls"><button id="pause" class="danger">Mettre en pause</button><button id="resume" class="primary">Reprendre</button></div></header>
<section class="panel"><div class="metrics"><div class="metric"><small>État de la tâche courante</small><strong id="state" class="state">…</strong></div><div class="metric"><small>Génération active</small><strong id="generation">…</strong></div><div class="metric"><small>Progression durable</small><strong id="steps">…</strong></div><div class="metric"><small>Checkpoints</small><strong id="checkpoints">…</strong></div><div class="metric"><small>Processus</small><strong id="processes">…</strong></div><div class="metric"><small>Dernier écart Elo</small><strong id="elo-gap">…</strong></div><div class="metric"><small>Cote Elo interne commune</small><strong id="internal-elo">…</strong></div></div><p><progress id="progress" max="100" value="0"></progress></p><p id="detail" class="note"></p><p id="elo-detail" class="note"></p><p id="action-status" class="note action-status"></p><p class="note">La consigne pause/reprise est écrite sur disque avant d'être appliquée. Après un redémarrage, une consigne « pause » reste en pause ; une consigne « exécuter » relance la même commande, qui repart de son dernier checkpoint validé.</p><pre id="log">Aucune sortie pour le moment.</pre></section>
<section id="live-metrics" class="panel" hidden><div class="metrics"><div class="metric"><small>Génération courante</small><strong id="live-current">—</strong></div><div class="metric"><small>Génération terminée</small><strong id="live-completed">—</strong></div><div class="metric"><small>Génération durable</small><strong id="live-durable">—</strong></div><div class="metric"><small>Parties collectées</small><strong id="live-games">—</strong></div><div class="metric"><small>Exemples entraînés</small><strong id="live-examples">—</strong></div></div><p class="note">Les cycles terminés en RAM peuvent dépasser la dernière génération durable. Le nombre de pas Adam peut diminuer après retour à un champion.</p></section>
<section id="teacher-panel" class="panel" hidden><h2>Accompagnement MCTS-32</h2><p id="teacher-stage"></p><p id="teacher-progress" class="note"></p><p id="teacher-evaluation" class="note"></p><p class="note">Objectif contre l’aléatoire : au moins 60 % de victoires et 20 % de nulles, ou au moins 80 % de victoires. Les évaluations indépendantes déterminent l’arrêt du programme.</p></section>
<div id="archive-head" class="archive-head"><p class="eyebrow">ARCHIVES DE LA CAMPAGNE PRINCIPALE</p><h2>Résultats générationnels</h2><p class="note">Cette section décrit uniquement les générations scellées de la campagne principale. Un diagnostic isolé peut travailler en haut de cette page sans augmenter ce compteur.</p></div>
{examples_link}<iframe id="dashboard" title="Archives et Elo de la campagne" src="/dashboard"></iframe>
<section class="panel" id="collection-panel" hidden><h2>Résultats des parties d’entraînement</h2><p id="collection-label">En attente de la prochaine collecte.</p><div class="metrics"><div class="metric"><small>Victoires</small><strong id="collection-wins">—</strong></div><div class="metric"><small>Défaites</small><strong id="collection-losses">—</strong></div><div class="metric"><small>Nulles</small><strong id="collection-draws">—</strong></div></div><p class="note">Parties terminales retenues pour apprendre, pas les matchs d’évaluation Elo. Mise à jour à la fin de la collecte de chaque génération.</p></section>
</main><script>
const token={token};let busy=false;let dashboardMtime=null;
const labels={{running:'En cours',paused:'En pause',completed:'Terminé',retrying:'Reprise automatique',starting:'Démarrage'}};
function number(v){{return new Intl.NumberFormat('fr-FR').format(v)}}
function signed(v){{return (v>=0?'+':'')+v.toLocaleString('fr-FR',{{minimumFractionDigits:1,maximumFractionDigits:1}})}}
async function refresh(){{try{{const r=await fetch('/api/status?'+Date.now(),{{cache:'no-store'}});const s=await r.json();
 const state=document.getElementById('state');state.textContent=labels[s.observed]||s.observed;state.className='state '+s.observed;
 const p=s.progress, target=p.target_step;document.getElementById('steps').textContent=target?number(p.completed_step)+' / '+number(target):number(p.completed_step);
 document.getElementById('checkpoints').textContent=number(p.checkpoint_count);document.getElementById('processes').textContent=s.pid?(number(s.usage.processes)+' · '+s.usage.cpu_percent.toFixed(1)+' % CPU'):'0';
 const m=s.campaign_metrics||{{}},latest=m.latest_evaluation,refs=m.rating_references||[],site=refs.find(x=>x.alias==='site-bot-v1');
 document.getElementById('teacher-panel').hidden=!m.teacher_program;
 if(m.teacher_program){{
 const phases={{'teacher-targets':'Préparation des cibles du professeur','teacher-learning':'Apprentissage avec le professeur','teacher-promotion':'Sélection du champion','teacher-evaluation':'Évaluation indépendante contre l’aléatoire',ppo:'Apprentissage PPO entre deux occurrences','goal-achieved':'Objectif atteint',starting:'Démarrage du programme'}};
 document.getElementById('teacher-stage').textContent=(phases[m.phase]||m.phase)+' · Occurrence '+(m.teacher_occurrence??'—')+' · Séquence '+(m.teacher_chunk??'—')+' / '+(m.teacher_chunks_per_occurrence??'—');
 document.getElementById('teacher-progress').textContent='Pas supervisés terminés : '+(m.teacher_steps_completed==null?'—':number(m.teacher_steps_completed))+' · Pas prévus par séquence : '+(m.teacher_steps_per_chunk==null?'—':number(m.teacher_steps_per_chunk));
 const evaluation=m.teacher_last_evaluation;
 document.getElementById('teacher-evaluation').textContent=evaluation?'Dernière évaluation indépendante : '+number(evaluation.wins)+' V / '+number(evaluation.draws)+' N / '+number(evaluation.losses)+' D sur '+number(evaluation.games)+' parties · '+(m.teacher_goal_passed?'Objectif atteint':'Objectif non atteint'):'Aucune évaluation indépendante du programme disponible.';
 }}
 document.getElementById('live-metrics').hidden=m.layout!=='paisho-live';
 document.getElementById('collection-panel').hidden=m.layout!=='paisho-live';
 const cr=m.collection_results;
 if(cr){{const pct=n=>cr.games?number(n)+' / '+number(cr.games)+' · '+(100*n/cr.games).toFixed(1)+' %':'—';
 document.getElementById('collection-label').textContent='G'+cr.generation+' · Adversaire : '+cr.opponent+' · '+number(cr.attempts)+' tentatives'+(cr.opponent==='self-play'?' · Auto-jeu : '+number(cr.self_play_decisive)+' parties décisives (une victoire et une défaite par partie).':'');
 document.getElementById('collection-wins').textContent=cr.opponent==='self-play'?'Deux sièges':pct(cr.wins);
 document.getElementById('collection-losses').textContent=cr.opponent==='self-play'?'Deux sièges':pct(cr.losses);
 document.getElementById('collection-draws').textContent=pct(cr.draws);}}
 document.getElementById('archive-head').hidden=m.layout==='paisho-live';document.getElementById('dashboard').hidden=m.layout==='paisho-live';
 if(m.layout==='paisho-live'){{for(const [id,key] of [['live-current','current_generation'],['live-completed','completed_generation'],['live-durable','durable_generation'],['live-games','games'],['live-examples','examples']]){{document.getElementById(id).textContent=m[key]===null||m[key]===undefined?'—':number(m[key]);}}}}
 document.getElementById('generation').textContent=m.active_generation===null?'—':'G'+number(m.active_generation);
 document.getElementById('elo-gap').textContent=latest&&latest.elo_gap!==null?signed(latest.elo_gap)+' Elo':'—';
 document.getElementById('internal-elo').textContent=m.internal_rating===null?'À mesurer':number(Math.round(m.internal_rating))+' Elo';
 document.getElementById('elo-detail').textContent=latest&&latest.elo_gap!==null?('G'+latest.generation+' contre '+latest.opponent+' : '+signed(latest.elo_gap)+' Elo relatif sur '+number(latest.eligible_pairs)+' paires. '+(site?('Repère commun : bot du site '+site.elo.toLocaleString('fr-FR',{{maximumFractionDigits:1}})+' Elo ; '):'')+'l’agent aléatoire doit encore être relié à cette échelle avant d’afficher une cote absolue du réseau.'):'Aucune évaluation générationnelle disponible.';
 document.getElementById('progress').value=p.percent||0;document.getElementById('detail').textContent=s.last_error||((s.pid?'PID '+s.pid+' · ':'')+(p.percent===null?'progression sans cible':p.percent.toFixed(1)+' % durable'));
 if(m.layout==='paisho-live'){{document.getElementById('detail').textContent+=' · Générations terminées : '+(m.completed_generation??'—')+' · Durables : '+(m.durable_generation??'—')+' · Parties : '+(m.games??'—')+' · Exemples : '+(m.examples??'—');if(latest&&latest.elo_gap===null)document.getElementById('elo-detail').textContent='G'+latest.generation+' contre '+latest.opponent+' : '+(latest.elo_kind||'écart indisponible');}}
 if(m.layout==='paisho-live'&&m.best_comparable_evaluation){{const best=m.best_comparable_evaluation;document.getElementById('elo-detail').textContent+=' · Meilleur écart comparable : G'+best.generation+' · '+signed(best.elo_gap)+' Elo';}}
 document.getElementById('log').textContent=s.log_tail||'Aucune sortie pour le moment.';document.getElementById('pause').disabled=busy||s.desired==='paused'||s.completed;document.getElementById('resume').disabled=busy||s.desired==='running'||s.completed;
 if(s.dashboard_mtime_ns!==null&&s.dashboard_mtime_ns!==dashboardMtime){{if(dashboardMtime!==null)document.getElementById('dashboard').src='/dashboard?'+s.dashboard_mtime_ns;dashboardMtime=s.dashboard_mtime_ns}}
 const actionStatus=document.getElementById('action-status');if(!busy&&s.completed){{actionStatus.textContent='Cette tâche précise est terminée. Le contrôleur attend qu’une nouvelle tâche lui soit confiée.'}}else if(!busy&&actionStatus.dataset.transient!=='yes'){{actionStatus.textContent=''}}
 }}catch(e){{document.getElementById('detail').textContent='Contrôleur momentanément inaccessible : '+e}}}}
async function action(name){{const status=document.getElementById('action-status');busy=true;status.dataset.transient='yes';status.textContent=name==='pause'?'Mise en pause demandée…':'Reprise demandée…';await refresh();try{{const r=await fetch('/api/'+name,{{method:'POST',headers:{{'X-Paisho-Control':token}}}});if(!r.ok)throw new Error('réponse HTTP '+r.status);const s=await r.json();status.textContent=name==='pause'?'Pause enregistrée sur disque.':'Reprise enregistrée sur disque.';setTimeout(()=>{{status.dataset.transient='no';refresh()}},2500)}}catch(e){{status.textContent='Action impossible : '+e;status.dataset.transient='yes'}}finally{{busy=false;await refresh()}}}}
document.getElementById('pause').onclick=()=>action('pause');document.getElementById('resume').onclick=()=>action('resume');refresh();setInterval(refresh,1000);
</script></body></html>""".encode("utf-8")


def make_handler(
    supervisor: Supervisor,
    dashboard: DashboardProvider,
) -> type[BaseHTTPRequestHandler]:
    config = supervisor.config
    gallery = (ArchiveGallery(config.dashboard.examples_archive_root)
               if config.dashboard and config.dashboard.examples_archive_root else
               Gallery(config.dashboard.examples_directory if config.dashboard else None))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", render_control_page(config))
            elif path == "/api/status":
                state = supervisor.status()
                state["examples_busy"] = dashboard.examples_busy
                payload = json.dumps(state, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, "application/json; charset=utf-8", payload)
            elif path == "/dashboard":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", dashboard.read())
            elif path == "/examples" or path.startswith("/examples/"):
                if path in ("/examples", "/examples/"):
                    dashboard.request_refresh()
                status, content_type, payload, download = gallery.response(path)
                self._send(status, content_type, payload, download)
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found\n")

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if not secrets.compare_digest(
                self.headers.get("X-Paisho-Control", ""),
                config.control_token,
            ):
                self._send(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"Forbidden\n")
                return
            if path not in {"/api/pause", "/api/resume"}:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found\n")
                return
            desired = "paused" if path.endswith("pause") else "running"
            payload = json.dumps(supervisor.set_desired(desired), ensure_ascii=False).encode("utf-8")
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", payload)

        def _send(self, status: HTTPStatus, content_type: str, payload: bytes, download: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(config: ControlConfig) -> None:
    config.working_directory.mkdir(parents=True, exist_ok=True)
    if not config.state_path.exists():
        atomic_write_json(config.state_path, default_state(config))
    supervisor = Supervisor(config)
    dashboard = DashboardProvider(config.dashboard, examples_allowed=lambda: load_state(config)["desired"] == "running")
    worker = threading.Thread(target=supervisor.run, name="training-supervisor", daemon=True)
    worker.start()
    server = ThreadingHTTPServer((config.host, config.port), make_handler(supervisor, dashboard))
    server.timeout = 0.5
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(f"control_url=http://{config.host}:{config.port}/", flush=True)
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        supervisor.stop()
        worker.join(timeout=2)
        server.server_close()


def set_desired_without_server(config: ControlConfig, desired: str) -> dict[str, Any]:
    if desired not in DESIRED_STATES:
        raise ControlError(f"invalid desired state {desired}")

    def update(state: dict[str, Any]) -> None:
        state["desired"] = desired
        if desired == "running":
            state["completed"] = False
        state["last_error"] = None

    return mutate_state(config, update)


def install_launch_agent(config: ControlConfig, label: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ControlError("launch-agent label contains unsupported characters")
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{label}.plist"
    service_log = config.log_path.with_name("control-service.log")
    payload = {
        "Label": label,
        "ProgramArguments": [
            sys.executable,
            str(Path(__file__).resolve()),
            "serve",
            "--config",
            str(config.path),
        ],
        "WorkingDirectory": str(config.working_directory),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 5,
        "StandardOutPath": str(service_log),
        "StandardErrorPath": str(service_log),
    }
    temporary = plist_path.with_name(f".{plist_path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        plistlib.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, plist_path)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["/bin/launchctl", "bootstrap", domain, str(plist_path)], check=True)
    subprocess.run(["/bin/launchctl", "enable", f"{domain}/{label}"], check=True)
    subprocess.run(
        ["/bin/launchctl", "kickstart", "-k", f"{domain}/{label}"],
        check=True,
    )
    return plist_path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("serve", "status", "pause", "resume", "new-token"):
        subparser = subparsers.add_parser(action)
        subparser.add_argument("--config", required=True, type=Path)
    install = subparsers.add_parser("install-launch-agent")
    install.add_argument("--config", required=True, type=Path)
    install.add_argument("--label", default="com.local.paisho.training-control")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    requested = sys.argv[1:] if arguments is None else arguments
    if requested and requested[0] == "gen5":
        if __package__:
            from .paisho_gen5_dashboard import main as gen5_main
        else:
            from paisho_gen5_dashboard import main as gen5_main
        gen5_main(requested[1:])
        return 0
    options = parse_arguments(arguments if arguments is not None else sys.argv[1:])
    config = ControlConfig.load(options.config)
    if options.action == "serve":
        serve(config)
    elif options.action == "status":
        print(json.dumps(Supervisor(config).status(), ensure_ascii=False, indent=2))
    elif options.action == "pause":
        print(json.dumps(set_desired_without_server(config, "paused"), indent=2))
    elif options.action == "resume":
        print(json.dumps(set_desired_without_server(config, "running"), indent=2))
    elif options.action == "new-token":
        print(secrets.token_hex(24))
    elif options.action == "install-launch-agent":
        print(install_launch_agent(config, options.label))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControlError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
