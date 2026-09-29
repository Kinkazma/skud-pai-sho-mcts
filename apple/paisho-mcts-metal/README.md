# Hybrid CPU / Metal MCTS corpus generator

The GPU evaluates heuristic features of batched candidates from independent
searches. The CPU retains rules, transitions, trees, UCT and individual leaf
handling. This is not a full GPU game engine, and a whole-game speedup was not
established by the original measurements.

```sh
python3 tools/paisho_mcts_corpus.py build
python3 tools/paisho_mcts_corpus.py generate runs/corpus --backend cpu --games 100
```

Inspect `--help` before a large run. Budgets 8/32/128/512 are search sizes of the
same engine, not rules variants. `--workers` controls simultaneous CPU games,
`--batch` GPU candidate batching and `--wait-us` optional batching delay.
CPU is a true reference mode, not a silent fallback for a failing GPU service.

The producer counts retained terminal games, archives seeds, sources, binaries,
attempt receipts and PSR records, and verifies hashes on continuation. Time-limited
attempts remain excluded. Errors stop the producer. The historical CPU32 deadline
was eight seconds; other modes use distinct defaults. An explicit deadline is
part of the experiment, not a speed-equivalence guarantee. Attempt limits prevent
unbounded retries. The frozen plan wins over later changed defaults on resume.

Every PSR contains the full game; cuts enumerate nonterminal Main-phase prefixes,
including the opening. Use these to construct populations without losing earlier
moves. Do not treat an interrupted game as a scored outcome.

This optional backend is preserved for research and has not been requalified by
the local CPU export. It does not change Gen3.5 training by being installed.
