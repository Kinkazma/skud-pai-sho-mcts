# Data and memory assets

The human data contains 1,375 normalized Skud V2 game records derived from a
larger tournament collection. The source collection had 1,724 downloaded games;
these are not 1,724 validated training games. Keep incomplete games marked as
incomplete. Normalized records contain game actions without player account names.
Derived features preserve split identities and held-out flags.

The code/weights MIT license does not grant rights in third-party source datasets.
Redistribution terms for the human corpus remain to be confirmed before public
upload. The local package includes it for preparation and testing; do not infer
permission merely from its prior availability online. No raw account profiles,
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
