# Data and memory assets

The human data contains 1,375 normalized Skud V2 game records derived from a
larger tournament collection. The source collection had 1,724 downloaded games;
these are not 1,724 validated training games. Keep incomplete games marked as
incomplete. Normalized records contain game actions without player account names.
Derived features preserve split identities and held-out flags.

The code/weights MIT license does not grant rights in third-party source datasets.
The project maintainer confirmed on 2026-09-29 that the human corpus and its
derived training data may be distributed for community reuse. Keep their source
provenance; this confirmation does not relicense third-party material as MIT.
No raw account profiles,
ratings, avatars or website assets are included.

- `human/records/*.psr`: normalized legal records.
- `human/records.json`: migration and split mapping with private paths removed.
- `assets/human/dataset.json.gz`: derived features with portable record paths.
- `assets/gen3.5-replay/`: relocalized FIFO/correction sources and index.
- `memory/`: binary sequence banks; weights refer to matching SHA-256 identities.
- `export-summary.json`: counts and scope of the completed local data extraction.

Large `.bin` and `assets/` data are not Git objects. They will accompany a matching
release with verified checksums; until then they are present in this local bundle.
The original metadata and machine paths are not needed to compute or train.

Install all parts of the groups listed in [the resource guide](ASSETS.md).
[The release asset catalog](release-assets.json) records archive names, sizes,
SHA-256 hashes and generation requirements. Large resources are not in a clone.
