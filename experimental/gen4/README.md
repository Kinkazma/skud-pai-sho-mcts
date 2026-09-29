# Historical CPU Micro Gen4

This is the archived 128→32 value/policy model (1,703,588 updates), distinct from
the Apple pure network and the later Gen5 architecture. This branch uses the
compatible historical CPU workspace. It is not a validated successor to Gen3.5.

From the repository root:

```sh
cargo build --release --locked -p paisho-train --bin paisho-micro
target/release/paisho-micro compare --model experimental/gen4/model.json --reference experimental/gen4/model.json --output runs/gen4-check --threads 1 --pairs 1 --seconds 15 --simulations 8 --decisions 8
target/release/paisho-micro selfplay --model experimental/gen4/model.json --output runs/gen4-new --threads 2 --workers 2 --seconds 60
```

Self-play starts from the archived weights; use `--replay` only with a compatible
Gen4 replay. This branch does not restore the entire historical campaign. Root
`manage.py` remains the Gen3 tool. The original model's numeric fields are
unchanged; private provenance paths were replaced with opaque artifact IDs.
