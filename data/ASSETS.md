# Release resources by generation

**A Git clone or GitHub's automatic source ZIP does not include these resources.**
Download every part of each group you need from the matching GitHub Release,
then extract the TAR files into the repository root. Do not put them in a nested
folder. Release upload and download URLs are still pending: this is a local draft.

## Which groups do I need?

| Use | Required groups | Optional groups |
| --- | --- | --- |
| Gen3.1–3.5 portable play/training | `core-memory` (one archive, about 196 MB) | `human` for human records/features |
| Gen3.5 with the historical `--replay` input | `core-memory` + `gen35-replay` (one archive, about 924 MB) | `human` |
| Experimental Gen5 portable launcher | `core-memory` + `human` + **all three** `gen5` archives | — |
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

The Gen5 group includes its memory bank, frozen FIFO, structured anchors, human
input mapping and starting archive subset. These files **do not constitute an
exact historical campaign resume**. The Gen5 guide documents the remaining limits.

## Install and verify

[release-assets.json](release-assets.json) is the machine-readable catalog. It
lists the exact filenames, byte sizes and SHA-256 digests of **every archive**,
with required groups by use. Each part is at most **1,800,000,000 bytes** (1.8 GB,
decimal), including TAR headers and padding, below the GitHub Release file limit.
The final part can be smaller. There is no need to download every generation.

```sh
# List filenames, sizes and hashes before obtaining the matching release assets.
python3 scripts/assets.py list --profile gen35-replay

# From the repository root, after downloading these files:
tar -xf /path/to/downloads/core-memory-01.tar
tar -xf /path/to/downloads/gen35-replay-01.tar
python3 scripts/assets.py verify --profile gen35-replay
python3 manage.py setup
```

For Gen5, install `core-memory-01.tar`, `human-01.tar`, `gen5-01.tar`,
`gen5-02.tar` **and** `gen5-03.tar`, then run:

```sh
python3 scripts/assets.py verify --profile gen5
python3 experimental/gen5/manage.py prepare
```

Run the Gen5 commands on `experimental/gen5`. Small tracked files in that branch
are also checked by the Gen5 manifest. The quick `check --profile NAME` command
checks presence and lengths; `verify` checks every file's SHA-256. Launchers fail
with the names of missing/incomplete groups before native training starts.
The release download step will be connected to real URLs only after publication.
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
