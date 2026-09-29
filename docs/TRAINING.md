# Training, models and branches

`main` and `gen3.5` use the Gen3.5 played model (26,390,182 updates) and the historical
residual-compatible trainer. `gen3.1`–`gen3.4` select older default weights; they
share the portable tools. These are weight-generation branches, not promises
that every past private campaign can be restarted exactly.

| Model | Updates | External memory |
|---|---:|---|
| 3.1 | 1,676,512 | none |
| 3.2 | 5,874,377 | gen3-sequences-v1 |
| 3.3 | 24,829,177 | gen3-sequences-v1 |
| 3.4 | 26,176,897 | gen34-sequences-v1-20260911 |
| 3.5 | 26,390,182 | gen34-sequences-v1-20260911 |

`crates/` is the compatible historical trainer. `inference/crates/` is the frozen
five-model inference suite. Do not replace the former with the latter: its older
Gen3 training artifact loader does not recognize the 3.5 residual schema.

## Gen3.1

The value-only compact model uses a different trainer:

```sh
cargo build --release --locked -p paisho-train --bin paisho-compact
python3 manage.py train --generation 3.1 --budget 8 --seconds 60 --workers 2 --output runs/new-g31
```

Its native `selfplay` command supports `--replay-input`; the policy-memory replay
pack of Gen3.5 must not be supplied to it. The compact `prepare` and `train`
commands construct and fit human-data features. Inspect the corresponding CLI
options in `crates/paisho-train/src/compact_learning/` before changing the dataset.

## Gen3.2–3.5

`paisho-gen32 run CONFIG` handles the compatible policy-memory lineage. It uses
SGD; there is no missing Adam state for this lineage. `human-fit` is a legacy
64-input fit and intentionally refuses 128-input Gen3.4/3.5 models. Use `run`
for their continued learning. The historical multi-budget recipe is preserved
in `configs/gen3.5-historical.json`; the launcher selects bounded resources and
one requested budget for a new experiment. Gen3.3 still has the 3.2 wire label.

The replay pack keeps FIFO order, correction indices and draw cursors. Only private
JSON string values are changed; numerical tokens in compressed source files are
preserved byte-for-byte. This is checked during extraction. Initial publication
of the data pack still requires a rights review for third-party human game data.

## Model installation and evidence

The source models under `models/` are immutable starting points. Training writes
new outputs under `runs/`, which Git ignores. No short smoke-test model replaces a
published generation. A model must be evaluated on independent, matched game
panels before any strength claim or installation on the public website.

The later search/feature/recall repairs in the development checkout were prepared
but not used to produce the historical 3.5 weights. Gen5 is separately experimental.

Gen3.1 uses the historical compact trainer. It has no cooperative pause request;
use a short explicit `--seconds` bound. The dashboard starts at Gen3.2.
Gen3.2–3.5 continuation reads `checkpoint.json`, the durable model/replay pair,
not the transient progress display.
