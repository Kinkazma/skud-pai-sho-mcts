"""Read paisho-live's small JSON summaries; never inspect checkpoints or replay data.

The controller's progress_directory is the live campaign root. Presence of
live-plan.json selects this adapter. Completed RAM cycles are informational;
only blocks/block-{generation:020}.json contribute durable training steps.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except OverflowError:
            pass
    return None


class LiveLayoutReader:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._collection_results: dict[str, Any] | None = None

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self._cache.get(path)
            if cached is not None and cached[0] == signature:
                return cached[1]
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            self._cache[path] = (signature, value)
            return value
        except (OSError, ValueError, UnicodeError):
            return {}

    def _paths(self, directory: str, prefix: str) -> list[Path]:
        return sorted(
            path for path in (self.root / directory).glob(f"{prefix}-*.json")
            if re.fullmatch(rf"{prefix}-\d{{20}}\.json", path.name)
        )

    def _assessment(self, generation: int) -> dict[str, Any]:
        return self._read(self.root / "assessments" / f"assessment-{generation:020}.json")

    @staticmethod
    def _belongs(assessment: dict[str, Any], block: dict[str, Any]) -> bool:
        return bool(block) and assessment.get("generation") == block.get("generation") and (
            assessment.get("candidate_sha256") == block.get("checkpoint_sha256")
        )

    @staticmethod
    def _protocol(options: dict[str, Any]) -> dict[str, Any]:
        # All observations come from this one immutable live plan. Per-generation
        # seeds deliberately vary and do not define a different evaluation protocol.
        return {
            key: value for key, value in options.items()
            if key.startswith("curriculum_") and "seed" not in key and key != "curriculum_directory"
        }

    def _evaluations(self, durable: int | None, options: dict[str, Any]) -> list[dict[str, Any]]:
        if durable is None:
            return []
        observations = []
        for path in reversed(self._paths("assessments", "assessment")):
            assessment = self._read(path)
            generation = _integer(assessment.get("generation"))
            if generation is None or generation > durable:
                continue
            analysis = assessment.get("evaluation")
            if not isinstance(analysis, dict):
                continue
            block = self._read(self.root / "blocks" / f"block-{generation:020}.json")
            if not self._belongs(assessment, block):
                continue
            point = analysis.get("contextual_elo_point") or {}
            estimate = (analysis.get("mle") or {}).get("candidate_minus_opponent") or {}
            interval = estimate.get("paired_cluster") or {}
            observations.append({
                "generation": generation,
                # Assessment.tier is AFTER evaluation; the block records its opponent.
                "opponent": block.get("tier"),
                "elo_gap": _number(point.get("elo")) if point.get("kind") == "finite" else None,
                "elo_kind": point.get("kind"),
                "davidson_gap": _number(estimate.get("estimate")),
                "ci95_low": _number(interval.get("interval_95_lower")),
                "ci95_high": _number(interval.get("interval_95_upper")),
                "eligible_pairs": _integer(analysis.get("eligible_pairs")),
                "conclusion": analysis.get("conclusion"),
                "checkpoint": assessment.get("selected"),
                "checkpoint_sha256": block.get("checkpoint_sha256")
                if assessment.get("selected") == block.get("checkpoint") else None,
                "protocol": self._protocol(options),
            })
        return observations

    def read(self) -> dict[str, Any] | None:
        if (self.root / "teacher-program-plan.json").is_file():
            return self._teacher_program()
        if not (self.root / "live-plan.json").is_file():
            return None
        plan = self._read(self.root / "live-plan.json")
        status = self._read(self.root / "live-status.json")
        if self._collection_results is None:
            self._collection_results = self._read(self.root / "collection-results-bootstrap.json") or None
        results = status.get("collection_results")
        if isinstance(results, dict):
            self._collection_results = results
        paths = self._paths("blocks", "block")
        block = self._read(paths[-1]) if paths else {}
        durable = _integer(block.get("generation"))
        # Before the first live block, status may describe the imported parent generation.
        if not paths:
            durable = _integer(status.get("durable_generation"))
        completed = _integer(status.get("completed_generation"))
        current = _integer(status.get("generation"))
        status_is_current = current is not None and (durable is None or current >= durable)
        completed = max(completed or 0, durable or 0) if completed is not None or durable is not None else None
        current = max(current or 0, completed or 0) if current is not None or completed is not None else None
        totals = status if status_is_current else block
        options = plan.get("options") or {}
        assessment = self._assessment(durable) if block and durable is not None else {}
        due = block and any(
            (interval := _integer(options.get(key))) and durable is not None and durable % interval == 0
            for key in ("promotion_every", "evaluation_every")
        )
        observations = self._evaluations(durable, options)
        latest = max(observations, key=lambda item: item["generation"], default=None)
        best = max(
            (item for item in observations if latest is not None
             and item["opponent"] == latest["opponent"]
             and item["protocol"] == latest["protocol"]
             and item["elo_gap"] is not None),
            key=lambda item: item["elo_gap"], default=None,
        )
        return {
            "layout": "paisho-live",
            "collection_results": self._collection_results,
            "active_generation": current,
            "current_generation": current,
            "completed_generation": completed,
            "durable_generation": durable,
            "phase": status.get("phase") if status_is_current else "durable",
            "games": _integer(totals.get("games")),
            "examples": _integer(totals.get("examples")),
            "durable_games": _integer(block.get("games")),
            "durable_examples": _integer(block.get("examples")),
            "completed_step": _integer(block.get("training_step")) or 0,
            "checkpoint_count": len(paths),
            "latest_checkpoint": block.get("checkpoint"),
            "assessment_pending": bool(due and not self._belongs(assessment, block)),
            "latest_evaluation": latest,
            "best_comparable_evaluation": best,
            "internal_rating": None,
            "internal_rating_status": "bridge-pending",
            "rating_references": [],
        }

    def _inherited_evaluation(self, checkpoint: Any, maximum_generation: int | None) -> dict[str, Any] | None:
        """Follow imported live checkpoints, reading only plans/assessment summaries."""
        seen = {self.root.resolve()}
        while isinstance(checkpoint, str) and checkpoint:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = self.root / path
            parent = next((p for p in path.parents if (p / "live-plan.json").is_file()), None)
            if parent is None or parent.resolve() in seen:
                return None
            seen.add(parent.resolve())
            reader = _reader(parent)
            plan = reader._read(parent / "live-plan.json")
            observations = reader._evaluations(maximum_generation, plan.get("options") or {})
            if observations:
                latest = max(observations, key=lambda item: item["generation"])
                return {**latest, "source_campaign": str(parent), "inherited": True}
            checkpoint = (plan.get("options") or {}).get("initial_checkpoint")
            imported = _integer(plan.get("initial_generation"))
            if imported is not None:
                maximum_generation = min(maximum_generation, imported) if maximum_generation is not None else imported
        return None

    def _teacher_program(self) -> dict[str, Any]:
        """Expose the supervisor's summary without opening model or replay files."""
        status = self._read(self.root / "program-status.json")
        generation = _integer(status.get("generation"))
        durable = _integer(status.get("durable_generation"))
        summary = {
            "layout": "paisho-live", "collection_results": None,
            "active_generation": generation, "current_generation": generation,
            "completed_generation": generation, "durable_generation": durable,
            "games": None, "examples": None, "durable_games": None,
            "durable_examples": None,
            "completed_step": _integer(status.get("completed_step")) or 0,
            "checkpoint_count": 1 if status.get("checkpoint") else 0,
            "latest_checkpoint": status.get("checkpoint"),
            "assessment_pending": False, "latest_evaluation": None,
            "best_comparable_evaluation": None, "internal_rating": None,
            "internal_rating_status": "bridge-pending", "rating_references": [],
        }
        active = status.get("active_live_directory")
        if status.get("phase") == "ppo" and isinstance(active, str) and active:
            directory = Path(active)
            if not directory.is_absolute():
                directory = self.root / directory
            # Forward only a real live campaign, never another program root.
            if (directory / "live-plan.json").is_file() and not (
                directory / "teacher-program-plan.json"
            ).is_file():
                child = _reader(directory).read() or {}
                # A fresh segment has no local blocks yet. Its zero counters must
                # not erase the durable checkpoint imported by the supervisor.
                if not child.get("latest_checkpoint"):
                    for key in ("completed_step", "checkpoint_count", "latest_checkpoint"):
                        child.pop(key, None)
                else:
                    child["checkpoint_count"] = child.get("checkpoint_count", 0) + (1 if status.get("checkpoint") else 0)
                summary.update(child)
        if summary.get("latest_evaluation") is None:
            inherited = self._inherited_evaluation(status.get("checkpoint"), generation)
            if inherited:
                summary["latest_evaluation"] = inherited

        summary.update({
            "teacher_program": True,
            "phase": status.get("phase", "starting"),
            "teacher_occurrence": _integer(status.get("occurrence")),
            "teacher_chunk": _integer(status.get("chunk")),
            "teacher_chunks_per_occurrence": _integer(status.get("chunks_per_occurrence")),
            "teacher_steps_per_chunk": _integer(status.get("steps_per_chunk")),
            "teacher_steps_completed": _integer(status.get("teacher_steps_completed")),
            "champion_checkpoint": status.get("champion_checkpoint"),
            "teacher_last_evaluation": status.get("last_evaluation")
            if isinstance(status.get("last_evaluation"), dict) else None,
            "teacher_goal_passed": status.get("goal_passed") is True,
        })
        return summary


@lru_cache(maxsize=8)
def _reader(root: Path) -> LiveLayoutReader:
    return LiveLayoutReader(root)


def read_live_layout(root: Path) -> dict[str, Any] | None:
    return _reader(root).read()
