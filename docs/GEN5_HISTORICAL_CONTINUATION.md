# Continue from the historical Gen5 stopping point

This is an optional, large-data workflow on `experimental/gen5`. The small
`gen5-play` and `gen5` profiles remain sufficient for comparison and a new pilot.
The historical workflow restores the later **learner separately from the
accepted actor**. It does not silently promote unaccepted weights.

## Algorithms

- `last-campaign`: the algorithm used in the final September 26 campaign, with
  portable file readers. This is the default for historical study.
- `prepared`: the September 27 draw-consolidation and structured-recall fixes.
  These were staged after the pause, but were not activated in another original
  production campaign. Choosing them starts an explicit modified experiment.

The six frozen source overlays for the former algorithm, their original hashes,
and the allowed I/O changes are under `experimental/gen5/history/last-campaign/`.
Opaque sequence-source digests retain memory-exclusion keys after private path
removal. Original bundle sort prefixes retain catalogue order. Lossless gzip
revision readers preserve numerical JSON tokens; new revisions supersede them
with the existing atomic JSON writer.

## Install, build and restore

Run from the repository root. Install **every** part of the selected profile;
the installer checks the member hashes and assembles the directories for you.
The complete historical archive needs substantial storage and the original
327,680-position FIFO accounts for about 32.55 GB in the native replay budget.
That is logical accounting, not a measured peak RSS; actual RAM also depends on
sharing, models, search trees and process overhead. This is not the portable 2 GiB pilot.

```sh
python3 scripts/assets.py install --profile gen5-history
python3 experimental/gen5/tools/build_history.py
cargo build --manifest-path portable-models/gen5-history-engine/Cargo.toml --release --locked -p paisho-train --example gen5_portable_recovery
python3 experimental/gen5/tools/continue_history.py prepare --mode last-campaign --seconds 3600 --output runs/history-study
python3 experimental/gen5/tools/continue_history.py verify --output runs/history-study
python3 experimental/gen5/tools/continue_history.py run --output runs/history-study
```

For the prepared fixes, build `experimental/gen5/Cargo.toml` instead, with both
`--bin paisho-gen5` and `--example gen5_portable_recovery`, then select
`--mode prepared` in a **different new output directory**.

Preparation creates an isolated writable archive using hard links on the same
filesystem. Creating the directory entries can take tens of minutes: the current archive
contains more than a million files. This preparation is separate from training.
The native writer replaces files atomically: installed archive contents remain
unchanged. Older archive roots are shared read-only. Filesystems without hard
links are not supported by this launcher; it fails rather than silently copying
tens of gigabytes. Keep the repository and the run together while it is active.
Do not edit linked files manually with an editor that writes in place.

No training runs during installation, preparation or verification. `run` has
the explicit new duration and refuses an already-started output. Loading counts
against that bound, so a very short duration can finish without learning.
The historical remaining duration is recorded for reference; it is never
automatically restarted. New models and game records stay in the new output.

## Verified state and practical limits

Original and portable restoration were compared for both algorithms: 327,680
FIFO positions, correction flags, numerical feature/target bits, the next 1,024
recall reads, protection state, acquisition registry and frozen evaluations.
The learner has 5,131,788 updates; the accepted actor remains separate.
Metadata identities necessarily change when their embedded paths change; the
comparison checks numerical state rather than asserting identical file hashes.

This preserves the saved durable state, not every in-flight worker, transient
cache or the identical future execution schedule. Hardware, asynchronous order
and wall-clock search deadlines can change future trajectories. The final fixes
and a successful short continuation do not establish that Gen5 learns durably
or becomes stronger. See [the data guide](GEN5_DATA_GUIDE.md) and
[release completeness](COMPLETENESS.md).

## Bounded continuation check — 30 September 2026

From the full saved state, the prepared engine processed 27 game receipts (25 terminal games) plus 35 reanalysis receipts and advanced SGD updates from 5,131,788 to 5,131,913, with no reported errors. The collection bound was 300 seconds including loading; native recorded elapsed time including drain/final persistence was 301.165 seconds, of which 199.918 seconds was initial loading. Process teardown is not a throughput measurement. No tested weights were promoted. Both algorithms passed original/portable state restoration; this subsequent training check used the prepared algorithm.

[Machine-readable verification](gen5-history-verification.json) records the exact scope and fingerprints. The transfer check is not a strength benchmark or a proof of durable Gen5 improvement.
