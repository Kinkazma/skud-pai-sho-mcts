from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    schema: int = 2
    rules: str = "skud-pai-sho-2022-03-14-v2"
    seed: int = 107
    width: int = 256
    layers: int = 6
    heads: int = 8
    feedforward: int = 1024
    context_moves: int = 12000
    representation: str = "absolute"
    environments: int = 8
    collection: str = "completion"
    length_median_factor: float = 4.0
    repetition_cycles: int = 4
    repetition_max_period: int = 128
    repetition_min_decisions: int = 24
    samples_per_game: int = 4096
    minibatch: int = 16
    epochs: int = 3
    learning_rate: float = 0.0001
    clip: float = 0.2
    target_kl: float = 0.02
    grad_clip: float = 1.0
    entropy_weight: float = 0.0
    baseline_decay: float = 0.9
    checkpoint_seconds: int = 120
    cpu_threads: int = 4
    history_limit: int = 256
    storage_limit_mb: int = 1024

    def validate(self):
        integers = [f.name for f in fields(self) if f.type is int]
        for name in integers:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.schema != 2 or self.rules not in (
            "skud-pai-sho-2022-03-14-v2", "skud-pai-sho-gen5-v1"
        ):
            raise ValueError("unsupported schema or rules")
        if self.collection not in ("completion","cohort"):
            raise ValueError("collection must be completion or cohort")
        if self.representation not in ("absolute","player-relative-v1"):
            raise ValueError("unsupported spatial representation")
        limits = {"width": (16, 1024), "layers": (1, 16), "heads": (1, 32),
                  "context_moves": (1, 12000),
                  "feedforward": (16, 4096), "environments": (1, 128),
                  "repetition_cycles": (2, 32), "repetition_max_period": (2, 512),
                  "repetition_min_decisions": (1, 4096), "samples_per_game": (1, 16384),
                  "minibatch": (1, 256), "epochs": (1, 20),
                  "checkpoint_seconds": (1, 3600), "cpu_threads": (1, 64),
                  "history_limit": (1, 4096), "storage_limit_mb": (256, 4096)}
        for name, (lo, hi) in limits.items():
            if not lo <= getattr(self, name) <= hi:
                raise ValueError(f"{name} must be in {lo}..{hi}")
        if self.width % self.heads or self.feedforward < self.width:
            raise ValueError("width must divide into heads; feedforward >= width")
        import math
        for name, lo, hi in (("learning_rate", 1e-7, 0.01), ("clip", 0.01, 0.5),
                             ("length_median_factor",2.0,20.0),
                             ("target_kl", 0.0001, 0.5), ("grad_clip", 0.01, 10),
                             ("entropy_weight", 0, 0.1), ("baseline_decay", 0, 0.999)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed out of range")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
