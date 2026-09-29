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
  **Do not run this template as a qualified resume**: its historical durable
  archives and relocated resume identities are not yet complete.
- `tools/`: historical diagnostics and controllers, preserved for further work.
  Not every legacy entry point has been requalified as a public launcher.

Run these from the repository root on this branch:

```sh
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

The full frozen FIFO is locally staged: 327,680 positions from 19,049 compressed
source files. `data-status.json` lists remaining exact-resume work. No experiment
started by these scripts writes back into the historical campaigns or promotes
its models to any played catalogue. Preparation/guard loading consumes part of
the explicit run duration, so very short bounds can produce no games.

## Status of the final fixes

Joint draw correction and structured-recall coverage preservation passed isolated
engineering checks in the original research. They were staged after the final
campaign pause. They have not been demonstrated to solve Gen5's long-term learning
problem in a subsequent production campaign. The historical sources/tests and
experimental state are supplied to make that investigation possible.
