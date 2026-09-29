# Skud Pai Sho — experimental Gen5

This branch preserves the unfinished Gen5 research. **No durable, convincing
improvement was demonstrated in production.** Gen3.5 remains the stable main
branch. The last accepted actor and the later learner are different artifacts.

Use [the Gen5 research guide](experimental/gen5/README.md) and its explicit
`experimental/gen5/manage.py` commands. The root `manage.py` remains the Gen3
compatibility tool. The historical Gen5 campaign resume is not yet qualified;
the supplied launcher creates a new bounded experiment from accepted weights.

The branch includes native sources, staged consolidation/recall repairs, model
weights, inputs and asset manifests. Human corpus redistribution was confirmed by the project maintainer.
Local source preparation does not constitute a public release or strength claim.

See [validation and remaining work](docs/VALIDATION.md),
[data packs](data/ASSETS.md), and [branch identities](docs/BRANCHES.md).

[Try the game on the public website](https://gaeldauchy.com/pai-sho/).

## Required release resources

Install `core-memory-01.tar`, `human-01.tar`, and **all three** Gen5 parts:
`gen5-01.tar`, `gen5-02.tar`, `gen5-03.tar`. They are not included in a Git clone
or the automatic source ZIP. See [installation and verification](data/ASSETS.md)
and [exact filenames, sizes and SHA-256 hashes](data/release-assets.json).
Run `python3 scripts/assets.py verify --profile gen5` before using the launcher.
