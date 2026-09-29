#!/usr/bin/env python3
"""Bounded game-level retention for sequence memory, without deleting source archives.

A refresh is transactional: it returns at most capacity games. Segment rows follow
membership, so deleting a generated game removes every indexed segment from it.
Human membership is immutable. Winning associations provide revocable rarity
protection; neither association nor this protection is a tactical proof.
"""
from collections import Counter
from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class Game:
    identity: str
    human: bool = False
    motifs: tuple[str, ...] = ()
    winning_motifs: frozenset[str] = field(default_factory=frozenset)
    recent_uses: int = 0
    useful_replays: int = 0
    last_added: int = 0

@dataclass(frozen=True)
class Policy:
    capacity: int = 50_000
    rare_sources: int = 3
    regular_uses: int = 8
    rare_fraction: float = .10


def select(existing, incoming, *, uses=None, policy=Policy()):
    """uses: motif retrieval counts over a *recent* window, not lifetime totals.

    Exact duplicates replace metadata only; human pinning can never be lost.
    Incoming data can be refused. Recency alone cannot evict a more useful case.
    The rare reserve is bounded to prevent rare wins from filling the bank.
    """
    if policy.capacity < 1 or not 0 <= policy.rare_fraction <= 1:
        raise ValueError('invalid sequence retention capacity')
    rows = {}
    for game in [*existing, *incoming]:
        old = rows.get(game.identity)
        if old and old.human and not game.human:
            continue
        if any(x < 0 for x in (game.recent_uses, game.useful_replays, game.last_added)):
            raise ValueError('negative retention evidence')
        rows[game.identity] = game
    human = [g for g in rows.values() if g.human]
    if len(human) > policy.capacity:
        raise ValueError('human corpus exceeds bank capacity; never evict humans')
    counts = Counter(m for g in rows.values() for m in set(g.motifs))
    uses = uses or {}
    agents = [g for g in rows.values() if not g.human]
    rare = {g.identity: frozenset(m for m in g.winning_motifs
            if m in g.motifs and counts[m] <= policy.rare_sources
            and uses.get(m, 0) <= policy.regular_uses) for g in agents}
    # Use source-level breadth and actual useful rehearsal. No unbounded lifetime
    # access counter: frequently reread old cases must still compete fairly.
    def score(g):
        diversity = sum(1 / counts[m] for m in set(g.motifs)) / max(1, len(set(g.motifs)))
        return (diversity * (1 + math.log1p(g.useful_replays) + .25 * math.log1p(g.recent_uses)),
                g.last_added, g.identity)
    slots = policy.capacity - len(human)
    reserve = min(slots, math.ceil(slots * policy.rare_fraction))
    selected = list(human)
    protected, represented = set(), set()
    for g in sorted(agents, key=score, reverse=True):
        fresh = rare[g.identity] - represented
        if fresh and len(protected) < reserve:
            selected.append(g);protected.add(g.identity);represented.update(fresh)
    # Discount neighborhoods already represented by pinned/protected exemplars.
    # The corpus-wide diversity score additionally penalizes redundant sources.
    represented = Counter(m for g in selected for m in set(g.motifs))
    rest = [g for g in agents if g.identity not in protected]
    rest.sort(key=lambda g: (sum(1/(1+represented[m]) for m in set(g.motifs)) / max(1,len(set(g.motifs))), *score(g)), reverse=True)
    selected.extend(rest[:slots-len(protected)])
    kept = {g.identity for g in selected}
    previous = {g.identity for g in existing}
    return selected, dict(kept=len(kept), added=sorted(kept-previous), evicted=sorted(previous-kept),
                         rejected=sorted({g.identity for g in incoming}-kept),
                         rare_protected=sorted(protected), human_pinned=len(human))
