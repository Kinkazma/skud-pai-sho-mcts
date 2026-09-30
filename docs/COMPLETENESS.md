# Release completeness and scope

This inventory distinguishes usable supplied models from an exact historical
campaign restoration. Resource groups and member hashes are versioned in
`data/release-assets.json` and `data/manifests/`. Website code/artwork and personal
hosting configuration are outside this export. No new model strength is claimed.

| Lineage | Engine and weights | Play / new training | Continuation and limits |
| --- | --- | --- | --- |
| Gen3.1 | Frozen CPU inference + compact value trainer; weights in `models/` | Root `manage.py`; no memory in this individual model | Native compact self-play accepts its final model and `--replay-input`; no cooperative pause |
| Gen3.2–3.4 | Frozen inference suite + compatible policy-memory trainer; weights in `models/` | Root `manage.py --generation`; common banks required | Native durable checkpoint/model/replay of new runs; not every old private campaign is exported |
| Gen3.5 (`main`) | Residual-compatible trainer and frozen inference; weights + two common banks | Play, serve, new training and portable controller | Durable continuation tested; historical 65,536-position FIFO + 128,718 corrections supplied separately |
| CPU Gen4 (`experimental/gen4`) | Historical micro engine and 1,703,588-update model | Native compare/selfplay commands in the branch guide | New training from weights; complete historical campaign schedule not exported |
| CPU Gen5 (`experimental/gen5`) | Native research workspace, accepted actor, separate later learner, fixes | Bounded compare and new experiment from accepted weights | Separate optional historical restoration: 327,680-position FIFO, full archives and state; original/prepared algorithms distinguished. See historical guide for verified scope and future-schedule limits |
| Apple MPSGraph | Swift service, Rust client, PPO/Adam, pure/micro network code; four pure checkpoints | Separate research tools, not a CPU Gen3/Gen5 accelerator | 54 Swift tests and checkpoint checksums passed; full historical Apple campaign continuation remains unqualified |
| Metal / Core ML | Corpus batching and optional model-conversion research sources | Separate experimental utilities | Core ML is inference-only; no missing PyTorch backend is claimed to have been recovered |

## Data actually supplied

- 1,375 normalized human PSRs and split/migration information; derived Gen3 dataset.
  The larger source download count was 1,724, not 1,724 validated training games.
  Incomplete records remain marked. Account profiles and participant identities
  are not required and are not exported. The maintainer confirmed distribution
  for community reuse; the corpus is not automatically relicensed as MIT.
- Gen3.5 frozen FIFO and correction sources, with relocalized indices/checksums.
- Gen5 bank, 19,049 frozen FIFO source files, structured anchors, progress snapshot,
  small pilot archive and human input mapping. The separate optional full-history
  group adds all 2,984,151 durable archive files and saved recall/protection state.
- Four Apple pure checkpoints: candidate and retained champion from two experiments.
  A candidate's presence does not imply promotion.

## Deliberate boundaries

The branch defaults select weight generations using compatible exported tools;
they are not copies of private Git history. Numeric weights and frozen engines
retain their identities, while private provenance/path strings were replaced.
No full collection of every intermediate checkpoint, exploratory campaign ledger,
or all former dashboard charts is asserted. The supplied English controller is
for launching, observing, pausing and continuing supported new runs.

The frozen release weights are never overwritten by training. Outputs are written
to new ignored directories. Continued learning is supported where documented;
progress, strength and bitwise identity across hardware are not guaranteed.
macOS checks are local; Linux CI is configured, and Windows supervision is not
qualified. See [validation evidence](VALIDATION.md) and [installation](INSTALL.md).
