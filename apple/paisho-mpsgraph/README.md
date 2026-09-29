# MPSGraph research backend

This macOS Swift package implements a separate policy/value network family. It
is not a drop-in implementation of the Gen3.5 CPU model or the later CPU Gen5.

- `pureV1`: residual trunk 10×160, hierarchical policy, WDL head, 4,698,679 parameters.
- `microV1`: trunk 3×20, policy embedding 8, value head 32, 29,307 parameters.
- StateEncodingV1: 17×17×29 NHWC planes and 26 global values.
- ActionEncodingV1: family/tile/source/destination addresses, legal-action masking.

The pure model performs search-free inference. The Rust MPSGraph client provides
binary framing, batching and checkpoint coordination. SwiftPM requires Swift 6;
Apple APIs require macOS. See `Package.swift` for the declared platform minimum.

```sh
sdk_path=$(xcrun --sdk macosx --show-sdk-path)
SDKROOT="$sdk_path" swift test --package-path apple/paisho-mpsgraph --sdk "$sdk_path"
SDKROOT="$sdk_path" swift build --package-path apple/paisho-mpsgraph --sdk "$sdk_path" -c release --product paisho-mpsgraph-service
```

## Execution and wire formats

A specialized `MPSGraphExecutable` and shared Metal buffers are reused per fixed
shape. Serving returns policy and value probabilities; the training/test path
also materializes logits. Training invalidates compiled inference so the next
execution sees the updated weights. Pipelined in-flight requests and CPU
prefetching remain experimental; one in-flight request is the default.

The service reserves stdout for binary protocols; diagnostics use stderr:

- PSI V1: inference.
- PST V1: historical supervised objective.
- PST V2: terminal PPO, clipped importance ratios for the behavior policy, terminal
  WDL targets, entropy and the frozen actor's baseline.
- PSC V1: immutable checkpoint publication and checksum response.
- PSG1: framed JSON corpus transition without reloading weights/Adam.

Steps are bound to snapshot identity, replay index range and expected step.
`--checkpoint PATH` loads a matching `--preset` into the requested batch/action
shape. The default checkpoint mode `resume` keeps strict identity/index/rate
checks. `new-generation` retains weights, Adam moments and global step but starts
a newly adopted replay stream at index zero.

## Checkpoint semantics

Checkpoints contain weights, Adam moments/velocities, training and scheduler state,
generation, replay index, named RNG states, network/batch/compiler configuration,
neural schema and rules profile. SHA-256 covers the contents. V2 binds replay
indices to an exact replay snapshot hash. Conflicting writes are rejected; an
identical PSC publication can be retried after a lost response.

Existing tests cover exact continuation of the next Adam step, PPO clipping,
win/loss signals, masked padding, extreme temperatures and finite importance ratios.
Those historical tests are source evidence; this export does not claim new Apple
hardware qualification. MPSGraph Level1 permits device placement but does not
promise ANE execution. Core ML is a separate inference-only study.
