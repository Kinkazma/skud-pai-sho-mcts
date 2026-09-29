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

## Remaining release work

- Human corpus distribution was confirmed by the project maintainer on 2026-09-29;
  it is not automatically relicensed as MIT.
- Finish the full historical Gen5 durable-archive/identity migration if an exact
  campaign continuation is required; the current experimental launcher starts anew.
- Reconcile/export older campaign schedules and archival reference histories beyond
  the provided model/replay snapshots; do not advertise every old run as restorable.
- Exercise Linux CI and document Windows limits. Apple tests pass, but a complete
  historical Apple campaign continuation remains unqualified.
- The English dashboard is a working portable controller, not yet the full old
  analysis dashboard with every chart and publication detail.
- Create actual GitHub releases, upload verified assets and set their real URLs
  only after the local-only instruction changes. The downloader is implemented
  and tested locally; no remote URL is fabricated.

Historical CPU Gen4 also built and passed a two-seat, eight-decision comparison
from its own archived weights on `experimental/gen4`. This is a load check only.

## Generation resource archives (2026-09-29)

Seven TAR files were prepared locally, each at most 1,800,000,000 bytes including
headers/padding. Every archive hash and all 67,343 contained files were verified
against the per-file manifests. Gen5 has three parts; Gen3.5 replay and common
banks each have one. Separate human and Apple groups remain optional by use.
Six packaging tests cover exact size bounds with long/Unicode paths, deterministic
metadata, tampering, missing parts, oversize members and overwrite refusal.
See [the resource catalog](../data/release-assets.json). No assets were uploaded.
