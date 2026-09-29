# Backends are separate model families

| Backend | Role | Relationship to Gen3.5 |
|---|---|---|
| Native Rust CPU | Gen3 inference and SGD training | used by main |
| Swift MPSGraph + Rust client | separate policy/value models, PPO and Adam checkpointing | research code, not a drop-in Gen3.5 accelerator |
| Metal batched heuristic evaluation | hybrid CPU/GPU corpus generation | experimental, no whole-game speedup established |
| Core ML MIL reconstruction | inference conversion and device experiments | no training/Adam export; FP16 can change actions |

The MPSGraph package includes `pureV1` (4,698,679 parameters) and `microV1`
(29,307 parameters), which are not the CPU Gen5 relational/memory model.
Their checkpoint formats are distinct. The Core ML study uses coremltools 8.3.0
and NumPy 1.26.4 and reconstructs MIL directly, without requiring PyTorch.
No PyTorch backend or weight conversion is claimed without a recovered implementation.

On macOS with Xcode/Swift, optional tests/builds are explicit:

```sh
sdk_path=$(xcrun --sdk macosx --show-sdk-path)
SDKROOT="$sdk_path" swift test --package-path apple/paisho-mpsgraph --sdk "$sdk_path"
SDKROOT="$sdk_path" swift build --package-path apple/paisho-mpsgraph --sdk "$sdk_path" -c release --product paisho-mpsgraph-service
```

These are not run automatically by CPU setup. Historical Apple checkpoint inventory
and end-to-end export qualification remain separate from the verified Gen3.5 path.

The exported Swift MPSGraph test suite passed all 54 tests locally with explicit
SDKROOT and `--sdk` as shown in its README. A first invocation without those
settings failed to resolve system headers; the documented invocation succeeded.
Four historical pure-network checkpoints are staged in `assets/apple-checkpoints`.
Both their internal trailing checksums and independent whole-file SHA-256 values
were verified. Loading every historical checkpoint through a full training
continuation has not been tested by this extraction. See `apple/checkpoints.json`.
