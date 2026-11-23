import json
import errno
from pathlib import Path
import signal
import time
import torch
from .config import Config
from .engine import Engine, engine_identity
from .model import TransformerPlayer, choose_device
from .ppo import update
from . import rollout, storage, budget
from .runtime import StageTimes, defaults


class Trainer:
    def __init__(self, config, engine, device):
        self.config, self.engine, self.device = config.validate(), engine, choose_device(device)
        selected = defaults(self.device, torch.__version__)
        torch.set_num_threads(selected["cpu_threads"] or config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        self.timings = StageTimes()
        torch.manual_seed(config.seed)
        self.model = TransformerPlayer(config, engine.description["points"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.sampling_rng = torch.Generator().manual_seed(config.seed+1)
        self.shuffle_rng = torch.Generator().manual_seed(config.seed+2)
        self.reservoir_rng = torch.Generator().manual_seed(config.seed+3)
        self.baseline = 0.0
        self.pending = None
        self.behavior = {}
        self.ready = []
        self.outliers = {"seen":0,"lengths":[],"median":None,"p99":None}
        self.counters = {"rollouts": 0, "updates": 0, "optimizer_steps": 0,
                         "started_games": 0, "finished_games": 0, "loop_stops": 0,
                         "blocked_stops": 0, "host_wins": 0, "guest_wins": 0, "draws": 0,
                         "length_outlier_stops":0,
                         "decisions_played": 0, "trained_decisions": 0, "elapsed_seconds": 0.0}
        self._clock = None
        self.history = []
        self.training_stage = "initialized"
        self.pretraining = None
        self.checkpoint_status = None

    def account_time(self):
        if self._clock is not None:
            now = time.monotonic()
            self.counters["elapsed_seconds"] += now-self._clock
            self._clock = now

    def record_closed_games(self, previous_done):
        # A loop stop closes a slot, but never counts as a completed game.
        for slot, (before, after) in enumerate(zip(previous_done, self.pending["done"])):
            if before or not after:
                continue
            outcome = self.pending["outcomes"][slot]
            if outcome != -2:
                self.counters["finished_games"] += 1
                category = {-1: "draws", 0: "host_wins", 1: "guest_wins"}[outcome]
                self.counters[category] += 1
            elif self.pending["terminations"][slot] == "loop-unknown":
                self.counters["loop_stops"] += 1
            elif self.pending["terminations"][slot] == "length-outlier-unknown":
                pass  # stream.collect atomically reserves its strict allowance.
            else:
                self.counters["blocked_stops"] += 1

    def payload(self):
        self.account_time()
        return {"schema": 3, "config": self.config.as_dict(),
                "engine_identity": engine_identity(), "source_identity": storage.source_identity(),
                "points": self.engine.description["points"],
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "sampling_rng": self.sampling_rng.get_state(), "shuffle_rng": self.shuffle_rng.get_state(),
                "reservoir_rng": self.reservoir_rng.get_state(),
                "torch_rng": torch.get_rng_state(), "baseline": self.baseline,
                "pending": self.pending,
                "behavior":self.behavior,"ready":self.ready,"outliers":self.outliers,
                "games": self.engine.request("export") if self.pending is not None else None,
                "counters": self.counters, "history": self.history,
                "training_stage": self.training_stage, "pretraining": self.pretraining}

    def summary(self):
        from .pretrain import summary as pretraining_summary
        self.account_time()
        pending = self.pending
        seconds = self.counters["elapsed_seconds"]
        return {"config": self.config.as_dict(), "parameters": self.model.parameter_count,
                "device": str(self.device), "counters": self.counters,
                "training_stage": self.training_stage, "pretraining": pretraining_summary(self.pretraining),
                "finished_games_per_second": self.counters["finished_games"]/seconds if seconds else 0.0,
                "phase": "collect-update" if self.ready else "ready" if pending is None else "update" if all(pending["done"]) else "collect",
                "ready_games":len(self.ready),"active_behavior_versions":len(self.behavior),
                "length_control":{"observed_finished_games":self.outliers["seen"],
                                  "median":self.outliers["median"],"p99":self.outliers["p99"],
                                  "median_factor":self.config.length_median_factor,
                                  "last_censor":self.outliers.get("last_censor"),
                                  "abandoned_fraction":self.counters["length_outlier_stops"]/max(
                                      1,self.counters["finished_games"]+self.counters["length_outlier_stops"])},
                "active_decisions": sum(pending["decisions"]) if pending else 0,
                "active_retained_samples": sum(len(t) for t in pending["trajectories"]) if pending else 0,
                "active_finished_games": sum(o != -2 for o in pending["outcomes"]) if pending else 0,
                "active_loop_stops": pending["terminations"].count("loop-unknown") if pending else 0,
                "active_optimizer_steps": self.ready[0]["progress"].get("steps",0) if self.ready else pending.get("ppo_progress", {}).get("steps", 0) if pending else 0,
                "last_rollout": self.history[-1] if self.history else None,
                "checkpoint_status":self.checkpoint_status,
                "performance": {"stages":self.timings.values, "native_requests":self.engine.timings,
                                "runtime":{"cpu_threads":torch.get_num_threads(),
                                           "engine_workers":self.engine.workers,
                                           "prefetch":defaults()["prefetch"],
                                           "deterministic":torch.are_deterministic_algorithms_enabled()}}}

    def save(self, directory):
        try:
            result = self.timings.call("checkpoint", storage.save, directory, self.payload(), self.summary(), self.config.storage_limit_mb)
            self.checkpoint_status = {"saved":True,"generation":result['generation'],
                                      "storage_reclamation":result['storage_reclamation']}
            return result
        except (budget.SpaceUnavailable, OSError) as error:
            if isinstance(error,OSError) and error.errno not in (errno.ENOSPC,errno.EDQUOT):
                raise
            self.checkpoint_status = {"saved":False,"deferred":True,"reason":str(error)}
            print(json.dumps({"checkpoint":self.checkpoint_status}),flush=True)
            return self.checkpoint_status

    @classmethod
    def restore(cls, directory, engine, device):
        saved, metadata, recovered = storage.load(directory)
        if saved["engine_identity"] != engine_identity():
            raise ValueError("checkpoint uses different rules source")
        if saved["source_identity"] != storage.source_identity():
            raise ValueError("training source changed; use a new run or an explicit migration")
        if saved["points"] != engine.description["points"]:
            raise ValueError("checkpoint board encoding differs")
        trainer = cls(Config.from_dict(saved["config"]), engine, device)
        trainer.model.load_state_dict(saved["model"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.sampling_rng.set_state(saved["sampling_rng"])
        trainer.shuffle_rng.set_state(saved["shuffle_rng"])
        trainer.reservoir_rng.set_state(saved["reservoir_rng"])
        torch.set_rng_state(saved["torch_rng"])
        trainer.baseline, trainer.pending = saved["baseline"], saved["pending"]
        trainer.counters, trainer.history = saved["counters"], saved["history"]
        trainer.counters.setdefault("length_outlier_stops",0)
        trainer.behavior = {v:{k:t.to(trainer.device) for k,t in p.items()} for v,p in saved.get("behavior",{}).items()}
        trainer.ready = saved.get("ready",[])
        trainer.outliers = saved.get("outliers",trainer.outliers)
        trainer.training_stage, trainer.pretraining = saved["training_stage"], saved["pretraining"]
        if trainer.pending is not None:
            observations = engine.request("restore", games=saved["games"])
            if observations != trainer.pending["observations"]:
                raise ValueError("restored native games differ from checkpoint observations")
            if any(o["rules"] != trainer.config.rules for o in observations):
                raise ValueError("restored rules differ from configuration")
        if recovered:
            # Commit the recovered state before rotating another slot.
            storage.atomic_json(Path(directory) / "state.json", metadata)
            print(json.dumps({"recovered_previous_checkpoint": metadata["generation"]}), flush=True)
        return trainer

    def run(self, directory, seconds, max_rollouts=None):
        if not 0 < seconds <= 7*24*3600:
            raise ValueError("seconds must be positive and at most seven days")
        self.start_ppo()
        deadline, last_save = time.monotonic()+seconds, time.monotonic()
        self._clock = time.monotonic()
        initial_rollouts = self.counters["rollouts"]
        stop_requested = False

        def stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in handlers:
            signal.signal(s, stop)
        try:
            while time.monotonic() < deadline and not stop_requested:
                if max_rollouts is not None and self.counters["rollouts"]-initial_rollouts >= max_rollouts:
                    break
                if self.config.collection == "completion":
                    from .stream import tick
                    previous_updates = self.counters["rollouts"]
                    metrics = tick(self,deadline,lambda:stop_requested)
                    if self.counters["rollouts"] != previous_updates:
                        print(json.dumps(self.history[-1],allow_nan=False),flush=True)
                    if metrics is not None and metrics.get("budget_stop"):
                        break
                    if time.monotonic()-last_save >= self.config.checkpoint_seconds:
                        self.save(directory)
                        last_save=time.monotonic()
                    continue
                if self.pending is None:
                    self.pending = rollout.start(self.engine, self.config, self.sampling_rng)
                    self.counters["started_games"] += self.config.environments
                if not all(self.pending["done"]):
                    previous_done = self.pending["done"].copy()
                    active = sum(not done for done in previous_done)
                    self.timings.call("selfplay_round", rollout.advance, self.engine, self.model, self.pending, self.config,
                                    self.sampling_rng, self.device, self.reservoir_rng)
                    self.counters["decisions_played"] += active
                    # Count real terminal results immediately, even if other
                    # slots are long-running and PPO has not updated yet.
                    self.record_closed_games(previous_done)
                else:
                    samples = rollout.finished_samples(self.pending)
                    progress = self.pending.setdefault("ppo_progress", {})
                    previous_steps = progress.get("steps", 0)
                    previous_unique = progress.get("unique_decisions", 0)
                    metrics, baseline = self.timings.call("ppo_update", update, self.model, self.optimizer, samples, self.config,
                                               self.baseline, self.shuffle_rng, self.device, deadline,
                                               lambda: stop_requested, progress, max_steps_per_call=32)
                    self.counters["optimizer_steps"] += metrics["optimizer_steps"]-previous_steps
                    self.counters["trained_decisions"] += metrics["unique_decisions_trained"]-previous_unique
                    if not metrics["complete"]:
                        if metrics.get("budget_stop") or stop_requested:
                            break
                        if time.monotonic()-last_save >= self.config.checkpoint_seconds:
                            self.save(directory)
                            last_save = time.monotonic()
                        continue
                    self.baseline = baseline
                    stats = rollout.statistics(self.pending)
                    self.counters["rollouts"] += 1
                    if metrics["updated"]:
                        self.counters["updates"] += 1
                    self.account_time()
                    rate = self.counters["finished_games"]/max(self.counters["elapsed_seconds"], 1e-9)
                    self.history.append({"rollout": self.counters["rollouts"], **stats, "ppo": metrics,
                                         "finished_games_per_second": rate})
                    self.history = self.history[-self.config.history_limit:]
                    print(json.dumps(self.history[-1], allow_nan=False), flush=True)
                    self.pending = None
                    self.save(directory)
                    last_save = time.monotonic()
                if time.monotonic()-last_save >= self.config.checkpoint_seconds:
                    self.save(directory)
                    last_save = time.monotonic()
        finally:
            try:
                self.account_time()
                self._clock = None
                self.save(directory)
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
        print(json.dumps(self.summary(), allow_nan=False), flush=True)

    def start_ppo(self):
        if self.training_stage == "ppo":
            return
        if self.training_stage != "pretrain" or self.pretraining is None or not self.pretraining["steps"]:
            raise ValueError("run human pretraining first: train.py pretrain --run ... --seconds ...")
        # Keep every learned model tensor; PPO gets optimizer moments suited to
        # its own objective rather than inheriting imitation-gradient momentum.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.baseline = 0.0
        self.training_stage = "ppo"


def initialize(directory, config, device):
    with storage.run_lock(directory):
        if any(p.name != ".lock" for p in Path(directory).iterdir()):
            raise ValueError("initialize requires an empty run directory")
        with Engine() as engine:
            trainer = Trainer(config, engine, device)
            trainer.save(directory)
            return trainer.summary()


def train(directory, seconds, device, max_rollouts=None):
    with storage.run_lock(directory), Engine() as engine:
        trainer = Trainer.restore(directory, engine, device)
        trainer.run(directory, seconds, max_rollouts)
