# Gen5 — experimental, not a validated successor

Gen5 introduced a larger value network, action-level policy/value coupling,
neural memory, structured motif learning, durable recall, and guarded publication.
The production experiments did **not** demonstrate durable, convincing overall
progress. Local acquisition/retention checks are not a general strength claim.

## Contents

- `crates/`: current native workspace, including the staged draw-consolidation
  and signed structured-recall repairs.
- `models/accepted.json`: last accepted actor, 5,124,160 updates.
- `models/learner.json`: later learner, 5,131,788 updates; never implicitly promoted.
- `inputs/`: five fixed opponents, human dataset mapping and protection panels.
- `configs/historical-template.json`: sanitized historical settings for study.
  Use the separate historical launcher below for a full-state restoration;
  this template alone is not a complete resume.
- `tools/`: historical diagnostics and controllers, preserved for further work.
  Not every legacy entry point has been requalified as a public launcher.

Run these from the repository root on this branch:

```sh
python3 scripts/assets.py install --profile gen5 --from-dir /path/to/downloads
python3 experimental/gen5/manage.py build
python3 experimental/gen5/manage.py compare --budget 8 --seconds 30 --decisions 8 --output runs/gen5-load-check
python3 experimental/gen5/manage.py train --budget 256 --workers 2 --seconds 120 --output runs/gen5-new
```

The comparison is a bounded load/decision test against Gen3.1; eight decisions do
not measure strength. The training command starts a **new experiment** from the
accepted weights, the existing protection panels, 128 structured anchors and a
fixed small archive subset (64 bundles and 64 proof files). It preserves the
50% recall setting and the repaired mechanisms, but does **not** restore the
historical FIFO, curriculum clocks, proof coverage epochs, or whole durable
archive. Its smaller 8,192-position/2 GiB replay is explicitly a portable pilot
configuration, not a reproduction of the 327,680-position/32 GiB campaign.

The optional full-state archive includes 327,680 FIFO positions from 19,049
compressed source files, the full durable archive and the reconciled resume state.
See [the historical continuation guide](../../docs/GEN5_HISTORICAL_CONTINUATION.md)
for restoration, separate learner/actor state and the choice of original algorithm
or staged fixes. `data-status.json` records verification scope. No experiment
started by these scripts writes back into the historical campaigns or promotes
its models to any played catalogue. Preparation/guard loading consumes part of
the explicit run duration, so very short bounds can produce no games.

## Status of the final fixes

Joint draw correction and structured-recall coverage preservation passed isolated
engineering checks in the original research. They were staged after the final
campaign pause. They have not been demonstrated to solve Gen5's long-term learning
problem in a subsequent production campaign. The historical sources/tests and
experimental state are supplied to make that investigation possible.

## Download only the resources needed

The `gen5-play` profile needs the common banks and Gen5 bank (about 583 MB).
The `gen5` new-training profile adds input mappings, seed lessons, structured
anchors and human records (about 638 MB total). The historical FIFO and full
archive are **optional**, for the separate historical restoration workflow.
See [what the data contains](../../docs/GEN5_DATA_GUIDE.md).

The matching release supplies `core-memory-01.tar`, `gen5-memory-01.tar`, plus
`human-01.tar` and `gen5-learning-01.tar` for training. Extract every required part into the repository
root. [The resource catalog](../../data/release-assets.json) records exact sizes
and hashes; [installation instructions](../../data/ASSETS.md) explain verification.
The verified installer accepts local archives or an explicit HTTPS release base URL.
The default URL remains unset until publication.
The launcher checks missing/incomplete groups before preparing models or training.
