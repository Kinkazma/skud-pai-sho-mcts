"""Small raw-record corpus and streaming, rules-replayed next-action examples."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from .engine import ROOT, Engine, engine_identity
from . import budget, storage

MANIFEST = ROOT / "data/human/manifest.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source_index, validation_fraction=0.2, seed=107):
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation fraction must be between zero and one half")
    source_index = Path(source_index).resolve()
    rows = json.loads(source_index.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("human source index must be a nonempty list")
    with budget.project_write_guard(), Engine() as engine:
        if MANIFEST.exists():
            existing = HumanDataset()
            if (existing.data["source_index_sha256"] != sha(source_index.read_bytes())
                    or existing.data["seed"] != seed
                    or existing.data["validation_fraction"] != validation_fraction):
                raise ValueError("the prepared corpus is immutable; source and split recipe must match")
            return existing.summary()
        budget.check(reserve=sum(((source_index.parent/r["record_path"]).stat().st_size+65535)//65536*65536
                                 for r in rows)+4*1024*1024)
        (ROOT / "data").mkdir(exist_ok=True)
        staging = ROOT / "data/.human-staging"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "records").mkdir(parents=True)
        entries, excluded, duplicates = [], [], 0
        parent, first_group = {}, {}
        def find(group):
            parent.setdefault(group, group)
            if parent[group] != group:
                parent[group] = find(parent[group])
            return parent[group]
        try:
            for offset in range(0, len(rows), 32):
                chunk, contents = rows[offset:offset+32], []
                for row in chunk:
                    path = (source_index.parent/row["record_path"]).resolve()
                    if not path.is_relative_to(source_index.parent):
                        raise ValueError("source record path escapes its corpus directory")
                    content = path.read_bytes()
                    digest = sha(content)
                    if digest != row["target_record_sha256"]:
                        raise ValueError(f"source checksum mismatch: {path.name}")
                    contents.append(content)
                facts = engine.request("audit_records", records=[c.decode() for c in contents])
                for row, content, fact in zip(chunk, contents, facts):
                    digest = sha(content)
                    if not fact["valid"] or fact.get("rules") != "skud-pai-sho-2022-03-14-v2" or not fact.get("decisions"):
                        excluded.append({"sha256": digest, "reason": fact.get("error", "wrong rules or empty record")})
                        continue
                    group = row.get("split_identity_sha256", digest)
                    find(group)
                    if digest in first_group:
                        a, b = find(group), find(first_group[digest])
                        parent[max(a, b)] = min(a, b)
                        duplicates += 1
                        continue
                    first_group[digest] = group
                    (staging / "records" / f"{digest}.psr").write_bytes(content)
                    entries.append({"file": f"records/{digest}.psr", "sha256": digest,
                                    "group": group, "decisions": fact["decisions"], "outcome": fact["outcome"]})
            for entry in entries:
                entry["group"] = find(entry["group"])
            groups = sorted({e["group"] for e in entries}, key=lambda group: sha(f"{seed}:{group}".encode()))
            if len(groups) < 2:
                raise ValueError("at least two independent games are required")
            validation = set(groups[:max(1, round(len(groups)*validation_fraction))])
            for entry in entries:
                entry["split"] = "validation" if entry["group"] in validation else "train"
            manifest = {"schema": 1, "rules": "skud-pai-sho-2022-03-14-v2",
                        "engine_identity": engine_identity(), "source_index_sha256": sha(source_index.read_bytes()),
                        "seed": seed, "validation_fraction": validation_fraction,
                        "entries": entries, "excluded": excluded, "duplicates": duplicates}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            staging.rename(MANIFEST.parent)
            budget.check()
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return HumanDataset().summary()


class HumanDataset:
    def __init__(self, manifest=MANIFEST):
        self.path = Path(manifest).resolve()
        if not self.path.is_relative_to(ROOT):
            raise ValueError("prepared corpus must remain inside GPT-SKUD")
        self.data = json.loads(self.path.read_text())
        if self.data["schema"] != 1 or self.data["engine_identity"] != engine_identity():
            raise ValueError("prepared corpus uses a different rules engine")
        self.identity = sha(self.path.read_bytes())
        self.entries = self.data["entries"]
        self.splits = {name: [i for i, e in enumerate(self.entries) if e["split"] == name]
                       for name in ("train", "validation")}
        left = {self.entries[i]["group"] for i in self.splits["train"]}
        right = {self.entries[i]["group"] for i in self.splits["validation"]}
        if not left or not right or left & right:
            raise ValueError("human train/validation game groups overlap or are empty")

    def summary(self):
        return {"corpus_identity": self.identity, "rules": self.data["rules"],
                "games": len(self.entries), "decisions": sum(e["decisions"] for e in self.entries),
                "duplicates_removed": self.data["duplicates"], "excluded_games": len(self.data["excluded"]),
                "splits": {name: {"games": len(indices),
                                  "decisions": sum(self.entries[i]["decisions"] for i in indices)}
                           for name, indices in self.splits.items()}}

    def record(self, index):
        entry = self.entries[index]
        path = (self.path.parent / entry["file"]).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError("prepared record path escapes corpus")
        content = path.read_bytes()
        if sha(content) != entry["sha256"]:
            raise ValueError("prepared human record checksum mismatch")
        return content.decode()

    def fetch(self, refs, engine):
        grouped = defaultdict(list)
        for slot, (game, decision) in enumerate(refs):
            if not 0 <= decision < self.entries[game]["decisions"]:
                raise ValueError("human decision index out of range")
            grouped[game].append((decision, slot))
        samples = [None]*len(refs)
        chunks, selections = [], []
        for game, positions in grouped.items():
            positions.sort()
            while positions:
                start = positions[0][0]
                selected = [p for p in positions if p[0] < start+1024]
                positions = positions[len(selected):]
                chunks.append(dict(record=self.record(game), rules=self.data["rules"],
                                   start=start, count=selected[-1][0]-start+1))
                selections.append(selected)
        # Native independent replays use CPU workers; returned order is stable.
        for begin in range(0, len(chunks), 256):
            replies = engine.request("replay_batch", chunks=chunks[begin:begin+256])
            for reply, selected in zip(replies, selections[begin:begin+256]):
                lookup = {e["index"]: e for e in reply["examples"]}
                for decision, slot in selected:
                    example = lookup[decision]
                    observation = {**example["observation"], "history": reply["history"]}
                    samples[slot] = {"observation": observation, "target": example["target"]}
        return samples


def next_block(state, dataset, limit=256):
    """Persist compact references/cursors, never reconstructed position arrays."""
    refs = []
    order = state["game_order"]
    while state["game_cursor"] < len(order) and len(refs) < limit:
        game = order[state["game_cursor"]]
        remaining = dataset.entries[game]["decisions"]-state["move_cursor"]
        take = min(remaining, limit-len(refs))
        refs.extend([[game, move] for move in range(state["move_cursor"], state["move_cursor"]+take)])
        state["move_cursor"] += take
        if state["move_cursor"] == dataset.entries[game]["decisions"]:
            state["game_cursor"] += 1
            state["move_cursor"] = 0
    return refs
