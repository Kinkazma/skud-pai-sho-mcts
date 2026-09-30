# Skud Pai Sho — experimental Gen5

This branch preserves the unfinished Gen5 research. **No durable, convincing
improvement was demonstrated in production.** Gen3.5 remains the stable main
branch. The last accepted actor and the later learner are different artifacts.

Use [the Gen5 research guide](experimental/gen5/README.md) and its explicit
`experimental/gen5/manage.py` commands. The root `manage.py` remains the Gen3
compatibility tool. The small launcher creates a new bounded experiment from accepted weights.
A separate optional [historical workflow](docs/GEN5_HISTORICAL_CONTINUATION.md)
restores the saved learner, accepted actor, FIFO, recall and protections. The
original/prepared algorithms and the tested continuation scope are distinguished.

The branch includes native sources, staged consolidation/recall repairs, model
weights, inputs and asset manifests. Human corpus redistribution was confirmed by the project maintainer.
Execution and restoration checks are not evidence of playing strength.

See [validation and remaining work](docs/VALIDATION.md),
[data packs](data/ASSETS.md), and [branch identities](docs/BRANCHES.md).

[Try the game on the public website](https://gaeldauchy.com/pai-sho/).

## Required release resources

For play/comparison, use `--profile gen5-play` (about 583 MB of resources).
For the new-training pilot, use `--profile gen5` (about 638 MB total).
The much larger historical learning-state archive is **optional**, separately
packaged with independently verified restoration checks. It is not needed for either
of those two uses. [What the data contains](docs/GEN5_DATA_GUIDE.md) explains the
distinction. Resources are not included in a Git clone or the automatic source
ZIP. See [installation and verification](data/ASSETS.md)
and [exact filenames, sizes and SHA-256 hashes](data/release-assets.json).
Run `python3 scripts/assets.py verify --profile gen5` before using the launcher.

See [installation and resource recovery](docs/INSTALL.md) and
[the completeness inventory](docs/COMPLETENESS.md).
