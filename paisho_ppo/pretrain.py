"""Phase 1: predict observed human next actions from their past-only context."""
import json
import math
import signal
import time
import torch
from torch.nn import functional as F
from .human import HumanDataset, next_block
from .model import pack
from . import storage
from .engine import Engine
from .trainer import Trainer
from .runtime import Lookahead, defaults
from .execution import Execution


def summary(state):
    if state is None:
        return None
    return {"corpus_identity": state["corpus_identity"], "phase": state["phase"],
            "epochs_completed": state["epochs_completed"], "target_epochs": state["settings"]["epochs"],
            "optimizer_steps": state["steps"], "examples_trained": state["examples_trained"],
            "elapsed_seconds": state["elapsed_seconds"], "done": state["done"],
            "last_epoch": state["history"][-1] if state["history"] else None}


def initialize_state(trainer, dataset, epochs=3, batch_size=16, learning_rate=0.0001):
    if not 1 <= epochs <= 100 or not 1 <= batch_size <= 256 or not 1e-7 <= learning_rate <= 0.01:
        raise ValueError("invalid pretraining recipe")
    if trainer.training_stage == "ppo" or trainer.pending is not None:
        raise ValueError("human pretraining must precede PPO")
    if trainer.config.rules != dataset.data["rules"]:
        raise ValueError("human corpus and model rule identities differ")
    rng = torch.Generator().manual_seed(trainer.config.seed+4)
    order = dataset.splits["train"]
    shuffled = [order[i] for i in torch.randperm(len(order), generator=rng).tolist()]
    trainer.training_stage = "pretrain"
    for group in trainer.optimizer.param_groups:
        group["lr"] = learning_rate
    trainer.pretraining = {
        "corpus_identity": dataset.identity, "settings": {"epochs": epochs, "batch_size": batch_size,
                                                          "learning_rate": learning_rate, "buffer": 256},
        "epochs_completed": 0, "steps": 0, "examples_trained": 0, "elapsed_seconds": 0.0,
        "phase": "train", "done": False, "rng": rng.get_state(),
        "game_order": shuffled, "game_cursor": 0, "move_cursor": 0,
        "block_refs": [], "block_order": [], "block_cursor": 0,
        "loss_sum": 0.0, "correct": 0, "examples": 0, "validation": None, "history": []}


class Pretrainer:
    def __init__(self, trainer, dataset, prefetch=None, execution=None):
        self.trainer, self.dataset = trainer, dataset
        self.state = trainer.pretraining
        if self.state is None or self.state["corpus_identity"] != dataset.identity:
            raise ValueError("pretraining checkpoint uses a different human corpus")
        self.rng = torch.Generator()
        self.rng.set_state(self.state["rng"])
        self.cache_key, self.cache = None, None
        enabled = defaults()["prefetch"] if prefetch is None else prefetch
        self.parallel_metrics = enabled
        self.execution = Execution(trainer.device, enabled, execution)
        self.replay_ahead = Lookahead(enabled)
        self.pack_ahead = Lookahead(enabled)
        self.packed = None

    def close(self):
        self.pack_ahead.close()
        self.replay_ahead.close()

    def _block(self, cursor, shuffle):
        if cursor["block_cursor"] >= len(cursor["block_refs"]):
            cursor["block_refs"] = next_block(cursor, self.dataset, self.state["settings"]["buffer"])
            cursor["block_cursor"] = 0
            n = len(cursor["block_refs"])
            cursor["block_order"] = torch.randperm(n, generator=self.rng).tolist() if shuffle else list(range(n))
            self.state["rng"] = self.rng.get_state()
        if not cursor["block_refs"]:
            return None
        key = tuple(map(tuple, cursor["block_refs"]))
        if key != self.cache_key:
            self.cache = self.replay_ahead.get(key, self.dataset.fetch, cursor["block_refs"], self.trainer.engine)
            self.cache_key = key
            # Only deterministic record references are anticipated. The live
            # cursor and shuffle generator still advance in the original order.
            ahead = {k: cursor[k] for k in ("game_order", "game_cursor", "move_cursor")}
            refs = next_block(ahead, self.dataset, self.state["settings"]["buffer"])
            if refs:
                self.replay_ahead.submit(tuple(map(tuple, refs)), self.dataset.fetch, refs, self.trainer.engine)
        start = cursor["block_cursor"]
        size = self.state["settings"]["batch_size"]
        indices = cursor["block_order"][start:start+size]
        samples = [self.cache[i] for i in indices]
        self.packed = self.pack_ahead.get((key, tuple(indices)), pack,
                                        [s["observation"] for s in samples], "cpu", self.trainer.config.context_moves)
        following = cursor["block_order"][start+size:start+2*size]
        if following:
            self.pack_ahead.submit((key, tuple(following)), pack,
                                   [self.cache[i]["observation"] for i in following], "cpu", self.trainer.config.context_moves)
        return samples

    def _metrics(self, samples, train):
        trainer = self.trainer
        trainer.model.train(train)
        observations = [s["observation"] for s in samples]
        targets = torch.tensor([s["target"] for s in samples], dtype=torch.long, device=trainer.device)
        with torch.set_grad_enabled(train):
            prepared = self.packed if self.packed is not None else pack(observations, "cpu", trainer.config.context_moves)
            self.packed = None
            logits = trainer.model(self.execution.move(prepared))
            loss = F.cross_entropy(logits, targets)
            if train and trainer.device.type == "mps" and self.parallel_metrics:
                correct = (logits.argmax(-1)==targets).sum().to(loss.dtype)
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = self.execution.clip(trainer.model.parameters(),trainer.config.grad_clip,
                                           error_if_nonfinite=False)
                observed = torch.stack((loss.detach(),norm,correct)).detach().cpu().tolist()
                if not all(math.isfinite(x) for x in observed):
                    raise FloatingPointError("non-finite human loss or gradient")
                self.execution.step(trainer.optimizer)
                return observed[0]*len(samples),int(observed[2])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite human next-action loss")
            if train:
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.execution.clip(trainer.model.parameters(), trainer.config.grad_clip,
                                    error_if_nonfinite=True)
                self.execution.step(trainer.optimizer)
        return float(loss.detach())*len(samples), int((logits.argmax(-1)==targets).sum().detach())

    def step(self):
        state = self.state
        if state["done"]:
            return False
        if state["phase"] == "train":
            samples = self.trainer.timings.call("prepare_wait", self._block, state, True)
            if samples is None:
                state["phase"] = "validation"
                state["validation"] = {"game_order": self.dataset.splits["validation"],
                                       "game_cursor": 0, "move_cursor": 0,
                                       "block_refs": [], "block_order": [], "block_cursor": 0,
                                       "loss_sum": 0.0, "correct": 0, "examples": 0}
                return True
            loss, correct = self.trainer.timings.call("human_train", self._metrics, samples, True)
            state["block_cursor"] += len(samples)
            state["steps"] += 1
            state["examples_trained"] += len(samples)
            cursor = state
        else:
            cursor = state["validation"]
            samples = self.trainer.timings.call("prepare_wait", self._block, cursor, False)
            if samples is None:
                n, vn = state["examples"], cursor["examples"]
                row = {"epoch": state["epochs_completed"]+1,
                       "train_examples": n, "train_nll": state["loss_sum"]/n,
                       "train_top1": state["correct"]/n,
                       "validation_examples": vn, "validation_nll": cursor["loss_sum"]/vn,
                       "validation_top1": cursor["correct"]/vn, "validation_complete": True}
                state["history"].append(row)
                state["history"] = state["history"][-32:]
                state["epochs_completed"] += 1
                print(json.dumps({"human_pretraining": row}), flush=True)
                if state["epochs_completed"] >= state["settings"]["epochs"]:
                    state["done"] = True
                    return False
                order = self.dataset.splits["train"]
                state.update(phase="train", game_order=[order[i] for i in torch.randperm(len(order), generator=self.rng).tolist()],
                             game_cursor=0, move_cursor=0, block_refs=[], block_order=[], block_cursor=0,
                             loss_sum=0.0, correct=0, examples=0, validation=None, rng=self.rng.get_state())
                return True
            loss, correct = self.trainer.timings.call("human_validation", self._metrics, samples, False)
            cursor["block_cursor"] += len(samples)
        cursor["loss_sum"] += loss
        cursor["correct"] += correct
        cursor["examples"] += len(samples)
        return True

    def run(self, directory, seconds, max_steps=None):
        if not 0 < seconds <= 7*24*3600:
            raise ValueError("pretraining requires a positive duration of at most seven days")
        deadline, last_save, tick = time.monotonic()+seconds, time.monotonic(), time.monotonic()
        initial_steps = self.state["steps"]
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in handlers:
            signal.signal(s, stop)
        try:
            while not stopped and time.monotonic()<deadline and not self.state["done"]:
                if max_steps is not None and self.state["steps"]-initial_steps >= max_steps:
                    break
                self.step()
                now = time.monotonic()
                self.state["elapsed_seconds"] += now-tick
                tick = now
                if now-last_save >= self.trainer.config.checkpoint_seconds:
                    self.trainer.save(directory)
                    last_save = time.monotonic()
        finally:
            self.close()
            self.state["elapsed_seconds"] += time.monotonic()-tick
            try:
                self.trainer.save(directory)
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
        print(json.dumps({"pretraining": summary(self.state)}), flush=True)


def pretrain(directory, seconds, device="auto", epochs=None, batch_size=None, learning_rate=None, max_steps=None):
    with storage.run_lock(directory), Engine() as engine:
        trainer = Trainer.restore(directory, engine, device)
        if trainer.config.representation == "player-relative-v1":
            from .zones import ZoneDataset
            dataset = ZoneDataset()
        else:
            dataset = HumanDataset()
        if trainer.pretraining is None:
            initialize_state(trainer, dataset, 3 if epochs is None else epochs,
                             16 if batch_size is None else batch_size,
                             learning_rate if learning_rate is not None else 0.0001)
        else:
            if trainer.training_stage != "pretrain":
                raise ValueError("this run has already moved to PPO")
            settings = trainer.pretraining["settings"]
            if epochs is not None and epochs != settings["epochs"]:
                raise ValueError("resume keeps the original epoch target")
            if batch_size is not None and batch_size != settings["batch_size"]:
                raise ValueError("resume keeps the original minibatch size")
            if learning_rate is not None and learning_rate != settings["learning_rate"]:
                raise ValueError("resume keeps the original pretraining learning rate")
        worker = Pretrainer(trainer, dataset)
        try:
            worker.run(directory, seconds, max_steps)
        finally:
            worker.close()
