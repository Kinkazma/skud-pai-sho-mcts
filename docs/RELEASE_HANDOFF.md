# Publication handoff

Publication preparation is authorized. Create the actual repository and push the
reviewed generation branches; keep `main` on Gen3.5 and Gen5 experimental.
Use `data/release-assets.json` for the exact Release resource filenames, lengths
and SHA-256 hashes. The optional full history uses independent TAR parts at most
1,800,000,000 decimal bytes each. Every part is required only for its own group.

Do not upload a whole working directory, original private Git history, caches,
restored runs or private export journals. Publish tracked source plus the verified
resource packs. Set the real Release URL on maintained branches, refresh source
manifests, commit, and check installation from the actual public GitHub URLs.
Local transport tests do not prove live delivery. Never replace published bytes
under an existing asset version; use a new Release for changes.

See [completeness](COMPLETENESS.md), [installation](INSTALL.md), the
[data guide](GEN5_DATA_GUIDE.md), and [historical continuation](GEN5_HISTORICAL_CONTINUATION.md).
