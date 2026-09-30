# Install, play, train and continue

The matching resource URL is recorded in `data/release-assets.json`.
The installer downloads only the selected generation/profile.
Replace the example download directory with your own; quote paths containing spaces. All commands
below run from the repository root. No campaign starts during installation.

## Prerequisites

Use Python 3.10 or newer, Git, Rust/rustup and a C linker. The pinned Rust toolchain
is read from `rust-toolchain.toml`; Cargo installs the locked Rust dependencies.
On macOS, install Apple's command-line developer tools if the linker is absent.
On Linux, install your distribution's C build tools first. No Python packages,
GPU frameworks or paid services are needed for Gen3 CPU play and training.

Apple experiments need macOS, Xcode/Swift and their separate instructions in
[BACKENDS.md](BACKENDS.md). Windows process supervision is not qualified.

## Install resources and build Gen3

For automatic verified downloads from the matching GitHub Release:

```sh
python3 manage.py doctor
python3 manage.py setup
```

For already downloaded TAR files, add `--asset-dir /path/to/downloads`.
This verifies and installs the common banks, compiles the compatible trainer and
frozen inference engine, then prepares machine-local paths. A subsequent `setup`
reuses intact resources. It does not re-download or rewrite verified data.
The small numeric model files are already in Git.

For historical Gen3.5 replay, also install its resource group:

```sh
python3 manage.py install --resource-profile gen35-replay --asset-dir /path/to/downloads
```

For human records/features, use `--resource-profile human`. For optional Apple
checkpoints use `--resource-profile apple`. See [resource groups](../data/ASSETS.md)
for all required filenames, sizes and hashes. A Git clone and GitHub's automatic
source ZIP do not include these TAR files.

The HTTPS downloader uses the versioned catalog's `release_url`. A mirror URL
can be passed with `--asset-base-url` instead of `--asset-dir`. The base URL is the directory ending in the
release tag, before the archive filename. 
The downloader follows HTTPS redirects, verifies size and SHA-256, caches complete
archives, and resumes interrupted `.partial` downloads when the server supports
Range. If Range is ignored, it restarts that archive safely. Loopback HTTP is
accepted for local tests; other plain HTTP sources are refused.

## Play with supplied weights

```sh
python3 manage.py play --generation 3.5 --opponent 3.1 --budget 32 --output runs/example.psr
python3 manage.py serve --generation 3.5 --budget 32
```

`serve` expects the JSON-lines protocol documented in README. The existing website
is linked there for interactive board play. This repository does not package a
second website. Never count a decision-limited sample as a completed strength test.

## Train, save and continue

```sh
python3 manage.py train --generation 3.5 --budget 32 --workers 4 --seconds 600 --output runs/first
python3 manage.py train --generation 3.5 --budget 32 --workers 4 --seconds 600 --resume-from runs/first --output runs/continued
```

Wait until the first process has exited. Use a new output directory for the next
segment and keep the previous run available: its checkpoint references its saved
model and replay sources. This restores learned parameters and durable replay;
it does not replay the exact old thread/clock schedule. Keep generation and
training settings explicit. `--replay` imports the supplied historical Gen3.5
reserve on a new run and needs its additional archive and sufficient RAM.
At budget 8 add `--heuristic-reference` to both commands; trained references start
at budget 32. See [TRAINING.md](TRAINING.md) for the other lineages and limitations.

The English local training controller is started explicitly:

```sh
python3 manage.py dashboard --port 8770
```

## Experimental Gen5

Switch to `experimental/gen5` before installing its resources:

```sh
python3 scripts/assets.py install --profile gen5 --from-dir /path/to/downloads
python3 experimental/gen5/manage.py build
python3 experimental/gen5/manage.py prepare
```

This installs the common banks, human corpus, `gen5-memory` and `gen5-learning`
parts (about 638 MB total). `--profile gen5-play` needs only the banks for a
comparison (about 583 MB). Neither includes the historical FIFO or full archive.
The optional historical set uses separately numbered parts of at most 1.8 GB;
the installer reconstructs their common directory tree without concatenation.
The branch's
research guide provides its bounded play/training commands and distinguishes a
new experiment from the optional [historical continuation](GEN5_HISTORICAL_CONTINUATION.md).
The `gen5-history` profile automatically installs its separate file index before
the full archive parts. The index is also optional and is not stored in Git.
Installing the assets does not solve Gen5's unproven long-term learning behavior.

## Verification, interruption and recovery

```sh
python3 scripts/assets.py verify --profile gen3
python3 scripts/source_snapshot.py
python3 -m unittest discover -s tests -v
```

The installer verifies the entire selected group's archives and member hashes
before replacing any destination file in that group. Missing or altered archives,
unexpected members, symbolic links and traversal paths are rejected. Existing
files with different contents are preserved by default. To deliberately restore
release resources, use `python3 scripts/assets.py install --profile gen3 --from-dir /path/to/downloads --repair`. This affects only files in the
resource manifest, never trained outputs under `runs/`.

Installation commits files individually after group verification. If interrupted
during that final step, rerunning completes the missing files; previously valid
files are reused. Completed groups are kept if a later group fails. Downloads are
cached in `.cache/release-assets/`; local `--from-dir` installs read the existing
archives directly without copying them into that cache. Staging needs up to one
group's uncompressed size in temporary free disk space. Network installations also
retain the downloaded archive cache. No executable code is taken from the archives.
