# Last campaign versus prepared repairs

The last production campaign used the September 26 engine, identified in
`original-manifest.json` (native executable SHA-256
`8034b71dc0db5186664d939f04d4c532dcb57f76c4e237c2f0d7f041ea4dc7fa`).

Six source files changed before the September 27 repair was staged. Their original
versions are preserved here and independently match that frozen manifest.
`tools/build_history.py` assembles the last campaign algorithm in a new build
directory, adding only portable historical record readers. It does not activate
the staged consolidation or structured-retention fixes.

The normal experimental workspace contains those later fixes. They were tested
in isolated diagnostics but were never run in a subsequent production campaign.
Choosing that engine is a deliberate experiment, not a reproduction of the last
campaign algorithm. Both choices start from the same distinct learner/actor
checkpoint; neither promotes a private diagnostic model.

Portable readers retain original source exclusion IDs and archive ordering, verify
content hashes, and decompress revision JSON losslessly. They do not change MCTS
selection, numerical learning, acceptance thresholds or rule outcomes. Exact
continuation refers to the durable starting state; in-flight games/caches were not
persisted by the historical checkpoint, and parallel scheduling and wall-clock
search limits do not promise identical future game sequences across machines.
