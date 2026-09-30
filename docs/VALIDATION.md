# Local extraction checks — 29 September 2026

These are packaging and execution checks, not a new strength evaluation.

| Check | Result |
| --- | --- |
| Gen3.1–3.5 numeric models versus frozen export | Exact values; portable provenance/path metadata differs |
| Compatible Gen3.5 trainer + frozen inference workspace | Release builds passed |
| Native core tests | 13 passed |
| Gen3.5 residual integration test | Passed |
| Gen3.2/3.3/3.4/3.5 model inspection | All accepted by the compatible trainer |
| Frozen inference sample | Legal bounded Gen3.5/Gen3.1 decisions and PSR output |
| JSON-lines service | Two valid JSON response lines; diagnostics only on stderr |
| Gen3.5 training | 6,307 new examples; 26,390,182 → 26,421,717 updates; errors empty |
| Gen3.5 continuation | 943 new examples; 26,421,717 → 26,426,432 updates; errors empty |
| Historical replay restoration | 65,536 FIFO positions, 7,369,594,080 RAM bytes, five reference models resident; errors empty |
| Gen3.1 bounded training | 2,575 updates; 16 terminal games and 2 unresolved attempts; errors empty |
| English dashboard | Start, continuation and cooperative pause verified through the browser |
| MPSGraph Swift suite | 54 tests passed with explicit SDKROOT and --sdk |
| Four historical Apple checkpoint files | Internal content checksums and whole-file hashes verified |

The complete Gen3.5 replay export also includes 128,718 correction entries. The
restoration check is not a long training run with that entire reserve. Skud
continuation uses checkpoint.json as its durable authority; progress.json may
be temporarily incomplete. Thread interleaving and clock history are not restored
bit for bit.

The dashboard pause test stopped after 4.59 seconds of a 30-second run and kept
25.41 seconds in its native receipt. Continuing Skud creates a distinct run with
the newly requested duration; it is not an extension of the old deadline. The
previous generation/budget remain selected even if form selectors are changed.

Gen5's experimental export compiled and played a two-seat, eight-decision load
check. A new experiment from the accepted actor, with a small recall seed archive,
processed three receipts and 17 learner updates with no reported errors. The
45-second collection bound took 96.95 seconds including initialization, draining
and final protection/persistence. This is neither a hard process-time limit nor
a throughput benchmark, publication proof, exact historical resume or gain in
playing strength. No tested model was promoted into the supplied frozen models.

## Scope limits recorded during initial extraction

- Human corpus distribution was confirmed by the project maintainer on 2026-09-29;
  it is not automatically relicensed as MIT.
- The small Gen5 launcher starts a new experiment. The full durable-state transfer
  was subsequently completed separately; see the September 30 historical verification.
- Reconcile/export older campaign schedules and archival reference histories beyond
  the provided model/replay snapshots; do not advertise every old run as restorable.
- Exercise Linux CI and document Windows limits. Apple tests pass, but a complete
  historical Apple campaign continuation remains unqualified.
- The English dashboard is a working portable controller, not yet the full old
  analysis dashboard with every chart and publication detail.
- Publication was subsequently authorized. Live release delivery is checked separately
  from the local execution results listed here.

## Generation resource archives (2026-09-29)

Seven TAR files were prepared locally, each at most 1,800,000,000 bytes including
headers/padding. Every archive hash and all 67,343 contained files were verified
against the per-file manifests. Gen5 has three parts; Gen3.5 replay and common
banks each have one. Separate human and Apple groups remain optional by use.
Six packaging tests cover exact size bounds with long/Unicode paths, deterministic
metadata, tampering, missing parts, oversize members and overwrite refusal.
See [the resource catalog](../data/release-assets.json) for the final release set.
These initial archive counts predate the additional optional historical export.

## Fresh installation check — 29 September 2026

The repository was cloned into a new temporary directory, without source-tree
assets or prepared models. All generated artifacts remained in the private test
copy. No website, original campaign, frozen model or remote repository changed.

- The verified installer restored the common banks from TAR, and `manage.py setup`
  compiled both Rust workspaces. A full Gen3.5/Gen3.1 MCTS-8 sample ended legally
  after 26 decisions. This single game is an execution check, not a strength test.
- Short Gen3.1, 3.2, 3.3, 3.4 and 3.5 training segments all completed without errors.
- Gen3.5 learned 635 fresh positions, then 637 after restoring the exact durable
  model/replay pair; final reserve 1,272. Updates 26,390,182 → 26,393,357 → 26,396,542.
- Gen3.1 resumed the exact final-model hash and 992 replay examples; 4,960 initial
  updates, then 4,800 more. This is weight/replay continuation, not a clock replay.
- Historical Gen3.5, human, Apple and all three Gen5 resource archives installed
  correctly. Gen5 preparation and re-installation passed; already correct files
  were reused. No new Gen5 training run was needed for this installer check.
- 19 installer/launcher tests passed, including interrupted HTTP download/Range,
  cache reuse, missing/corrupt parts, path/symlink rejection, explicit repair and
  dotted output names. The check found and fixed the `gen3.2`/`gen3.3` sidecar-name
  collision before continuing the generation trials.
- The English dashboard HTML/API returned 200 in the fresh checkout and stayed idle.

These initial checks used macOS and loopback HTTP. Later live GitHub delivery
and CI results are reported separately; the versioned catalog now pins the real
Release URL.

See [machine-readable results](onboarding-results.json).

## Full Gen5 historical transfer — 30 September 2026

Original/portable native FIFO bits, the next 1,024 recall reads and restored protections/evaluations match for both historical and prepared algorithms. A relocated full-state prepared continuation also generates and learns new data. The continuation found and fixed missing nested opponent-bank paths before release. See [the exact scope and results](GEN5_HISTORICAL_CONTINUATION.md) and [machine-readable evidence](gen5-history-verification.json). No original campaign was restarted.

## GitHub release delivery

The final resource catalog records the actual GitHub Release URL and independently matched remote asset SHA-256 digests. The optional full-history TAR members were all reread against the per-file manifests locally. Live clone/install/play results are attached to the Release as `github-verification.json`; they distinguish actual downloads from the full-history remote digest checks. The large historical set is not downloaded a second time merely to repeat those hashes.
