# Local data packs

Large data stays outside Git objects. All paths are relative to this repository.
`data/manifests/*.json` records every file's byte length and SHA-256 digest.

| Group | Content | Status |
| --- | --- | --- |
| core-memory | Two Gen3 sequential banks | Required for Gen3.2–3.5 inference/training |
| human | Normalized human records and derived features | Local only; redistribution terms pending |
| gen35-replay | Complete exported Gen3.5 FIFO and correction sources | Optional historical training input |
| apple | Four checkpoints: candidate and retained champion in two experiments | Research; separate network and original V1 rules |
| gen5 | Gen5 memory, FIFO, structured anchors and small starting archive | Experimental branch; **not** an exact campaign-resume pack |

```sh
python3 scripts/assets.py verify core-memory
python3 scripts/assets.py pack core-memory --output packs
```

The pack command verifies all inputs, creates portable tar files of roughly
1 GiB of payload each, and records their checksums. It does not publish them.
Run separately for each desired group. Standard tar extracts files into the
checkout. Then run `python3 manage.py prepare` to regenerate machine-local paths.
Do not publish the human/derived groups before their redistribution terms are
resolved. No download URL exists yet. A later GitHub release will need matching
assets and an actual downloader URL; Git cloning alone does not carry these files.
