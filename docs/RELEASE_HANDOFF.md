# Publication handoff — not executed

The current instruction is to stop before creating or publishing GitHub repositories.
No remote URL, GitHub repository or Release is created by installation or tests.

After explicit publication authorization, the remaining operator actions are:

1. Create the actual repository and push the reviewed branches/tags. Keep `main`
   on Gen3.5, with Gen5 clearly experimental.
2. Attach the seven verified resource TAR files to the matching Release. Use
   `data/release-assets.json` as the exact filename/length/SHA-256 inventory.
   Do not upload the entire local working directory, build caches, runtime outputs,
   ignored prepared models, personal manifests or original private Git history.
3. Set the real Release asset directory in `data/release-assets.json:release_url`
   on the maintained branch views, refresh their source manifests, and commit.
4. Test the documented installation from the real downloadable source and assets,
   including HTTPS redirects, then a short game/training continuation. Local
   HTTP fixtures do not prove live GitHub distribution.

The TAR hashes are pinned independently of the transport location. Do not replace
an existing published archive with different bytes under the same version.
The public source/data inventory and known limits are in [COMPLETENESS.md](COMPLETENESS.md).
MIT applies to original code and owned weights; third-party provenance remains.
See [INSTALL.md](INSTALL.md) for user-facing installation commands.
