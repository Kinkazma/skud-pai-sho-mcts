# What the Gen5 data contains

**The complete historical archive is optional. It is not required to play with
the supplied model or to start new training from its weights.** It is intended
for contributors who want to restore the historical durable learning state or
audit how its targets were produced. The native restored numerical state has been compared with the original.
See [the historical workflow](GEN5_HISTORICAL_CONTINUATION.md) for the separate
continuation checks and limits.

## Choose a download scope

| Purpose | Profile | Historical archive needed? |
| --- | --- | --- |
| Load the accepted Gen5 model and run the bounded comparison | `gen5-play` | No |
| Start the documented new-training pilot from accepted weights | `gen5` | No; uses the supplied seed examples and anchors |
| Restore the learner, FIFO, ordered recall, protections and curriculum at the historical stopping point | `gen5-history` | Yes, for the supplied full-state restoration method |
| Study code, architecture, results and unresolved issues | Git source and documentation | No |

The new play resources contain the 387 MB Gen5 sequence bank and the 196 MB common
banks used by the reference opponents. New-training resources add approximately
45 MB of input mappings, seed lessons and structured anchors, plus the 9 MB human
corpus archive. Weights are tracked in Git. Exact download sizes and hashes are in
the release catalog. These scopes do not require the multi-gigabyte historical
FIFO or the complete durable archive. Use the matching [release assets](https://github.com/Kinkazma/skud-pai-sho-mcts/releases/tag/v0.1.0).

The older combined `gen5` resource group remains readable for compatibility.
Its three TAR files contain extra historical FIFO data; downloading them is no
longer the default for play or a new pilot. Already installed files are reusable.

## Three different kinds of historical record

**Ordinary bundles.** A compressed bundle stores a replayable game record or
continuation, its source/model identity, and one or more training lessons. A lesson
can contain the decision, policy target, value target, observed outcome, search
estimates, action values, tactical evidence and structured labels. Some bundles
come from reanalysis of existing positions. They are generated learning data,
not a collection of that many distinct human games.

**Proof records.** These store a replayable position prefix and a tree of legal
actions that certifies a win, loss or draw under the engine's rules. Verification
does not depend on trusting the neural network's score. They support exact
teaching targets and tactical recall. A proof count is not a count of distinct
learned skills, independent games or human demonstrations.

**Revisions.** These are position-indexed merged teaching records. They can keep
the currently selected lesson, the search estimate, the observed lesson,
separately proved information and outcomes from different continuations. This
separation prevents a later unproved estimate from simply erasing an observed
loss. It also repeats substantial arrays across fields. Structured annotations,
where present, describe action consequences such as harmony changes, rings,
exhaustion and known/unknown immediate threats. The revisions are not prose logs
or documentation, and not all their fields are structured motif labels.

## What the large numbers mean

The saved ordinary coverage manifest references **714,200 bundles**. The saved
proof coverage manifest references **227,584 proof files**. These are two lists
with saved traversal cursors, not two additive counts of unique training games.
The inventory found no missing referenced file. Other files were added after the
current coverage epoch began; the complete archive is therefore larger than
these two saved lists.

The modern archive's `revisions` directory alone contains **1,256,168 files** and
**84,738,099,116 bytes** of uncompressed JSON (84.74 GB decimal, 78.92 GiB).
Across all four archive roots, revisions occupy 90.65 GB, ordinary bundles
approximately 30.01 GB (already compressed), and proof files approximately
1.01 GB. These are source file payload sizes, not promised download sizes or
total installation requirements; the FIFO and other resources are additional.

An inspection of 128 modern revision files found 8,281,986 bytes of JSON becoming
1,910,796 bytes with lossless gzip level 1. The selected, estimated and observed
lesson fields accounted for approximately 98.6% of their JSON payload. This was
the first 128 regular filesystem entries, **not a representative random sample**;
do not extrapolate a final archive size or statistical redundancy rate from it.
The export compresses revisions without changing their numeric JSON tokens.

The completed export of **all 1,256,168 modern revision files** measures
**20,203,898,553 bytes** after compression and provenance relocation, compared
with 84,738,099,116 source bytes. This is a complete size measurement for that
directory, not the final size of the entire historical resource set. Other
archives, FIFO data, metadata and TAR packaging overhead are additional.

The complete archive export contains **2,984,151 files / 53,061,355,353 payload
bytes**, before the additional FIFO, small checkpoint dependencies and TAR
packaging overhead. Exact packaged sizes are recorded in the resource catalog.

## Relevance and limits

These records are read by the historical recall/reanalysis pipeline. Removing
them changes future samples or targets, even if the initial neural weights are
unchanged. That is why the full restoration option preserves them. This does
**not** establish that every record is useful, unique, correct as an estimate,
or responsible for improved strength. Proofs and estimates remain distinguished.
The poor historical learning outcome is not evidence that all collected data is
worthless, nor that retaining all of it will solve the learning problem.

A smaller curated training dataset may be useful in a new experiment, but it
must be labeled as a changed replay distribution. Lossless compression is the
first size reduction because it preserves the available information. Removing
duplicates, truncating history or retaining only selected proofs requires a
separate equivalence/coverage analysis before claiming the same durable state.

Restoring that state does not promise the identical future trajectory on another
machine: asynchronous scheduling and wall-clock limits can change future games.
Historical checkpoints also do not contain every in-flight game or transient
cache. Documentation remains available without downloading any historical data.
