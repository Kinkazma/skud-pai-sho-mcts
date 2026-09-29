# Branches and model identities

`main` and `gen3.5` select Gen3.5. `gen3.1` through `gen3.4` select their historical
model by default. These are portable release views with the compatible suite,
not copies of the old private repository history. Explicit `--generation` still
selects another included model. Gen3.3 keeps the historical wire value `3.2`.

`experimental/gen5` contains the current experimental CPU workspace, the last
accepted actor and the later private learner as different files, and the staged
consolidation/structured-recall fixes. No guarantee of strong or lasting learning
is made. Its new-experiment command defaults to the accepted actor.

The Apple `pure` experiments are a different family, with their original V1
rules and MPSGraph checkpoints. Their internal generation counters (3 and 63)
are **not** the public Scale generations or Skud Gen3.5. The early Micro Gen4
checkpoint is kept with the experimental workspace. No missing older generation
has been invented to make the numbering look continuous.

`experimental/gen4` provides the recovered CPU Micro Gen4 weights and native
play/self-play commands. Its bounded two-seat load check passed with no errors.
