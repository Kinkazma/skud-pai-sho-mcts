# Core ML inference study

A MIL reconstruction of the MPSGraph inference equations using checkpoint weights.
This does not export PPO, backpropagation, Adam state or an MPSGraph executable.
It uses coremltools 8.3.0 and NumPy 1.26.4 directly, without PyTorch or TensorFlow.

```sh
python3 -m venv apple/paisho-coreml-study/.venv
apple/paisho-coreml-study/.venv/bin/pip install -r apple/paisho-coreml-study/requirements.txt
apple/paisho-coreml-study/.venv/bin/python tools/coreml_export.py CHECKPOINT --inspect-only
apple/paisho-coreml-study/.venv/bin/python tools/coreml_export.py CHECKPOINT model.mlpackage --batch 1 --capacity 128
xcrun coremlcompiler compile model.mlpackage compiled/
```

Inspect the command help and supply an actual compatible MPSGraph checkpoint.
Gen3.5 JSON weights are not compatible. `InspectComputePlan.swift` needs macOS14.4+
and an already compiled model. Preferred placement is not a runtime activity trace.
`PredictFixture.swift` runs native packed fixtures for actual execution tracing.

The historical 64-position study reported maximum policy/WDL absolute errors of
2.98e-8 / 6.56e-7 for FP32, with no changed policy argmax. FP16 reported
2.74e-4 / 6.89e-3 and one changed policy argmax. Thus FP32 was numerically close
on that corpus, not bit-identical; FP16 was demonstrably an approximation.
Historical native tracing observed ANE inference with FP16. This does not prove
ANE training or faster end-to-end learning. No adoption is implied.

Fixtures contain flattened spatial/global features, action indices/presences,
legal masks and policy/WDL reference probabilities. Shapes must agree exactly;
indices remain valid even in padding slots. `coreml_compare.py` checks finite
outputs, errors, normalization, padding and argmax changes before optional timing.
Timing is opt-in and includes prediction/marshaling, not model load. The exported
source does not claim a fresh reproduction of the old hardware measurements.
