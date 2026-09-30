# Release resources by generation

**A Git clone or GitHub's automatic source ZIP does not include these resources.**
Use the verified installer with the matching
[GitHub Release](https://github.com/Kinkazma/skud-pai-sho-mcts/releases/tag/v0.1.0).
It downloads every required part and assembles their common directory tree.
Manual download followed by `--from-dir` is also supported.

## Which groups do I need?

| Use | Required groups | Optional groups |
| --- | --- | --- |
| Gen3.1–3.5 portable play/training | `core-memory` (one archive, about 196 MB) | `human` for human records/features |
| Gen3.5 with the historical `--replay` input | `core-memory` + `gen35-replay` (one archive, about 924 MB) | `human` |
| Experimental Gen5 bounded play/comparison | `core-memory` + `gen5-memory` (about 583 MB total) | Learning data |
| Experimental Gen5 new-training pilot | `core-memory` + `human` + `gen5-memory` + `gen5-learning` (about 638 MB total) | Full historical state, installed separately |
| Historical CPU Gen4, using its native commands | No additional resource group | Compatible new training inputs |
| Apple historical checkpoint experiments | `apple` (one archive, about 226 MB) | `human` if the experiment uses the corpus |

The root Gen3 launcher prepares the entire five-model suite and thus requires the
common banks even when selecting Gen3.1 (whose own model has no external memory).
Gen4/Gen5 use their respective experimental branches. Scale's profile weights
are small and are tracked directly in its separate repositories.

The `human` group is a separate archive of about 9 MB. Distribution of the corpus
and derived training data was confirmed by the project maintainer for community
reuse. Source provenance is retained; it is not automatically relicensed as MIT.
See [data provenance](README.md).

**The large Gen5 historical archive is optional**, for restoring/auditing the old
learning state. It is not required to play or start a new experiment from weights.
[The Gen5 data guide](../docs/GEN5_DATA_GUIDE.md) explains ordinary bundles, proofs,
revision records, their sizes and their limitations.

The optional `gen5-history` profile first installs the roughly 251 MB
`gen5-history-index` archive, then the history parts. The 301 compressed index
shards are pinned by a small tracked manifest; they do not enlarge an ordinary
Git clone. All history parts are required for this exact durable-state workflow.

The legacy combined `gen5` group includes its memory bank, frozen FIFO, structured anchors, human
input mapping and starting archive subset. These files **do not constitute an
exact historical campaign resume**. The Gen5 guide documents the remaining limits.

## Install and verify

[release-assets.json](release-assets.json) is the machine-readable catalog. It
lists the exact filenames, byte sizes and SHA-256 digests of **every archive**,
with required groups by use. Each part is at most **1,800,000,000 bytes** (1.8 GB,
decimal), including TAR headers and padding, below the GitHub Release file limit.
The final part can be smaller. There is no need to download every generation.

The preferred host is **GitHub Releases**, separate from Git source history.
GitHub's release documentation, checked on 2026-09-29, allows up to 1,000 assets
per release, each below 2 GiB, with no stated total release size or bandwidth
limit: [About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
Our 1.8 GB decimal cap is below that per-file limit. The full historical state is
an optional numbered archive set, not a default dependency of play/new training.

Parts are ordinary TAR archives containing different files under the same
relative directory tree. The installer checks every part and every member,
then assembles that tree automatically; no binary concatenation or special
multi-volume decompressor is needed. Compressed historical revision files are
read directly by the engine, so users need not expand all of them to raw JSON.

An external host such as Mega is a fallback. The same numbered files and SHA-256
manifest can be mirrored without changing the dataset. Download all required
parts to one folder and use `--from-dir`; a Mega sharing-page URL is **not** a
direct `--base-url` download endpoint. No Mega account or SDK is required by the
game engine or local installer.

```sh
# List filenames, sizes and hashes before obtaining the matching release assets.
python3 scripts/assets.py list --profile gen35-replay
# Verify and install downloaded archives automatically, without manual extraction.
python3 scripts/assets.py install --profile gen35-replay
python3 manage.py setup
```

The installer also supports `--base-url` for a real HTTPS release asset directory.
It verifies and reuses intact installed files and cached archives, resumes partial
downloads where supported, and verifies all members before installation. Missing
or corrupt parts leave that group's existing data untouched. See
[installation and recovery](../docs/INSTALL.md) for details.

For the Gen5 new-training pilot, the `gen5` profile selects the smaller dedicated
groups. The historical FIFO is not required:

```sh
python3 scripts/assets.py install --profile gen5 --from-dir /path/to/downloads
python3 experimental/gen5/manage.py prepare
```

For a comparison only, use `--profile gen5-play` and then the Gen5 `compare`
command. The old combined `gen5-01.tar` through `gen5-03.tar` remain compatible
under `--profile gen5-legacy-bundle`; they are not the default download.

Run the Gen5 commands on `experimental/gen5`. Small tracked files in that branch
are also checked by the Gen5 manifest. The quick `check --profile NAME` command
checks presence and lengths; `verify` checks every file's SHA-256. Launchers fail
with the names of missing/incomplete groups before native training starts.
The default URL and all expected hashes are pinned in the versioned catalog.
Moving a checkout requires running its preparation command again to regenerate
ignored local model paths; it does not require downloading intact assets again.

## Build or independently inspect release archives

`data/manifests/*.json` records each source asset's relative path, length and hash.
Archive contents are portable: no machine paths or local owner names are stored
in TAR metadata. Resource files stay outside Git objects.

```sh
python3 scripts/assets.py pack gen5 --output packs/new-release
python3 scripts/assets.py verify-packs gen5 --output packs/new-release
```

Packaging verifies inputs and refuses to overwrite an existing archive set.
Partitioning counts the final TAR bytes, including extended path headers and
end padding. A single source file too large to fit is rejected explicitly.
`verify-packs` rereads all members against the source manifest, checks archive
hashes and rejects missing, duplicate or unexpected members. It does not extract
or publish anything. Identical input manifests produce identical new archives.
The common archive prepared earlier is retained with its original verified hash.
