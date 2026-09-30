> This release branch defaults to **Generation 3.4**. Explicit generation arguments in the shared examples override that default. See `release.json`.

# Skud Pai Sho MCTS — Gen3.5

**Gen3.5 is the main, historically played model.** An `experimental/gen5` branch
preserves a more ambitious architecture and learning system intended to have
greater potential. Despite extensive work, Gen5 did **not** demonstrate durable,
convincing improvement. It is shared so others can investigate further, not as a
stronger or validated successor.

**[Try Pai Sho on the public website](https://gaeldauchy.com/pai-sho/).** The site's
Skud frontend is a separately maintained build and must not be assumed to use
the same weights as this repository. This project supplies engines, weights,
memories and training tools; it does not build a new website or WordPress ZIP.

## What is included

- Frozen inference models for Gen3.1–3.5 and their sequential memory banks.
- A compatible historical Gen3.5 trainer, including its 128→16→1 value residual.
- A separate frozen inference workspace for the five-model service.
- The Gen3.5 replay index: 65,536 FIFO positions and 128,718 corrections, backed
  by 46,773 relocalized source files in the optional data pack.
- 1,375 normalized human game records and the derived training dataset, with redistribution confirmed by the project maintainer.
- Optional Apple MPSGraph, Metal and Core ML research code, with distinct schemas.
- English setup, command-line and local training-management entry points.

The original export's trainer could not load the Gen3.5 residual schema. This
repository deliberately uses compatible training sources and retains the frozen
inference engine separately. Later staged spatial/search changes are not silently
substituted for the code that produced the historical model.

## Requirements and setup

Rust via rustup (the toolchain file pins 1.77.0), Python 3.10+, Git and a C linker.
Apple research requires macOS and Xcode/Swift as documented separately. Gen3.5
runs on CPU; installing an Apple backend does not accelerate it automatically.

```sh
python3 manage.py doctor
python3 manage.py setup
python3 manage.py play --generation 3.5 --opponent 3.1 --budget 32 --output runs/example.psr
python3 manage.py serve --generation 3.5 --budget 32
```

**`setup` downloads and verifies the [`core-memory` resource group](data/ASSETS.md).
Git clone and the automatic GitHub source ZIP do not include large data.**
For historical Gen3.5 replay, also install `gen35-replay`; for the human corpus,
install `human`. Gen5 and Apple have separate groups; see the generation table.
Use the [installation guide](docs/INSTALL.md) for automatic local/HTTPS resource
installation, verification and recovery. The versioned catalog pins the matching Release URL. `prepare` verifies
asset hashes and produces ignored, machine-local copies under `portable-models`.
Run it again after moving the checkout. Published models retain portable paths.

`serve` is a JSON-lines protocol. After `ready`, send `{"cmd":"start","setup_index":0}`,
`{"cmd":"choose"}` and `{"cmd":"apply","action":"..."}` for every actual move,
including the opponent's. `state`, `record` and `quit` are also available.

## Continue learning from the weights

```sh
python3 manage.py train --generation 3.5 --budget 32 --workers 4 --seconds 600 --output runs/new-g35
python3 manage.py pause --output runs/new-g35
python3 manage.py train --generation 3.5 --budget 32 --workers 4 --seconds 600 --resume-from runs/new-g35 --output runs/continued-g35
python3 manage.py dashboard --port 8770
```

Pause is cooperative. Wait for the process to exit before continuing. Each new
output is distinct. `--resume-from` restores the durable model and replay; it does
not claim a bit-identical restart of threads, clocks or the old campaign schedule.
Add `--replay` when starting from the *historical Gen3.5* replay pack. Its FIFO
alone was estimated at 6.86 GiB in RAM, plus memory banks, workers and overhead.
The launcher defaults to a configurable 12 GiB replay ceiling. A smaller ceiling is rejected at initial restoration; it does not silently
truncate the imported FIFO. Later incoming examples can cause ordinary eviction.

Training at budget 8 can use `--heuristic-reference` because the historical
trained reference pool begins at 32. This clears the trained pool and keeps the
trainer's historical heuristic fallback; it is not pure self-play. Read
[training and lineage notes](docs/TRAINING.md) before comparing experiments.

## Reproducibility and limitations

Generation, rules, search budget and solver setting are separate. Gen3.3's
historical artifact still says `generation: "3.2"`; the inference loader handles
that identity deliberately. Gen3.1/3.2 disable the solver in the frozen suite;
Gen3.3–3.5 enable it. The suite defaults to the original V2 rules, with an explicit
Gen5-rules mode available in its native CLI.

Code and numeric weights are preserved. Private provenance strings and resource
paths are rewritten, so public file hashes differ. Replay source hashes are
recomputed accordingly. Seeded inference is reproducible within a fixed runtime;
no cross-hardware CPU/GPU bitwise promise or guaranteed learning improvement is made.

The CPU extraction has local macOS smoke checks. Linux CI is provided for the
core sources; Windows and Apple training have not been requalified by this export.
An incomplete game is not counted as a regulatory win. A rising update counter
shows training execution, not increasing playing strength.

## License and data

Original project code and owned weights: [MIT](LICENSE). Third-party notices remain
applicable. The normalized human dataset is **not relicensed as MIT**; see
[data documentation](data/README.md). No personal hosting setup, credentials or
website artwork is part of the public source export.

See [installation](docs/INSTALL.md), [completeness by generation](docs/COMPLETENESS.md),
and [local validation and remaining work](docs/VALIDATION.md).
