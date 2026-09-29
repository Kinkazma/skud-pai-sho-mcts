"""Experimental inference-only MIL reconstruction; no model loading or prediction."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from coreml_checkpoint import read_checkpoint


def build_program(config, weights, batch, capacity, precision="float32", debug_logits=False):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if precision not in ("float32", "float16"):
        raise ValueError("unsupported precision")
    used = set()

    def w(name):
        used.add(name)
        return weights[name]

    def conv(x, name):
        # MPSGraph NHWC/HWIO -> MIL NCHW/OIHW, then restore NHWC.
        x = mb.transpose(x=x, perm=[0, 3, 1, 2])
        x = mb.conv(x=x, weight=w(name + "/weights").transpose(3, 2, 0, 1).copy(),
                    bias=w(name + "/bias"), pad_type="same")
        return mb.transpose(x=x, perm=[0, 2, 3, 1])

    def dense(x, name):
        return mb.add(x=mb.matmul(x=x, y=w(name + "/weights")), y=w(name + "/bias"))

    def norm(x, name):
        mean = mb.reduce_mean(x=x, axes=[3], keep_dims=True)
        centered = mb.sub(x=x, y=mean)
        variance = mb.reduce_mean(x=mb.mul(x=centered, y=centered), axes=[3], keep_dims=True)
        inverse = mb.rsqrt(x=mb.add(x=variance, y=np.float32(config["normalizationEpsilon"])))
        return mb.add(x=mb.mul(x=mb.mul(x=centered, y=inverse), y=w(name + "/gamma")),
                      y=w(name + "/beta"))

    specs = [mb.TensorSpec(shape=(batch, 17, 17, 29)), mb.TensorSpec(shape=(batch, 26))]
    specs += [mb.TensorSpec(shape=(batch, capacity), dtype=dtype) for dtype in
              (types.int32, types.int32, types.fp32, types.int32, types.fp32,
               types.int32, types.fp32, types.fp32)]

    @mb.program(input_specs=specs, opset_version=ct.target.macOS13)
    def program(spatial, global_features, family_indices, tile_indices, tile_presence,
                destination_indices, destination_presence, pair_indices, pair_presence, legal_mask):
        channels = config["trunkChannels"]
        trunk = conv(spatial, "stem/conv")
        globals_ = mb.reshape(x=dense(global_features, "stem/global"), shape=[batch, 1, 1, channels])
        trunk = mb.relu(x=norm(mb.add(x=trunk, y=globals_), "stem/norm"))
        for block in range(config["residualBlocks"]):
            prefix = f"trunk/block_{block}"
            residual = mb.relu(x=norm(conv(trunk, prefix + "/conv_1"), prefix + "/norm_1"))
            residual = norm(conv(residual, prefix + "/conv_2"), prefix + "/norm_2")
            trunk = mb.relu(x=mb.add(x=trunk, y=residual))
        pooled = mb.reduce_mean(x=trunk, axes=[1, 2], keep_dims=False)
        family = dense(pooled, "policy/family")
        tile = dense(pooled, "policy/tile")
        destination = mb.reshape(x=conv(trunk, "policy/destination"), shape=[batch, 289])
        embedding = config["policyEmbeddingChannels"]
        source = mb.reshape(x=conv(trunk, "policy/source_embedding"), shape=[batch, 289, embedding])
        target = mb.reshape(x=conv(trunk, "policy/destination_embedding"), shape=[batch, 289, embedding])
        pairs = mb.matmul(x=source, y=target, transpose_y=True)
        pairs = mb.mul(x=pairs, y=np.float32(1 / np.sqrt(embedding)))
        pairs = mb.reshape(x=pairs, shape=[batch, 289 * 289])
        logits = mb.gather_along_axis(x=family, indices=family_indices, axis=1)
        for scores, indices, presence in ((tile, tile_indices, tile_presence),
                (destination, destination_indices, destination_presence), (pairs, pair_indices, pair_presence)):
            part = mb.gather_along_axis(x=scores, indices=indices, axis=1)
            logits = mb.add(x=logits, y=mb.mul(x=part, y=presence))
        if precision == "float16":
            # Approximation: replace invalid logits, rather than adding -1e9.
            # -1e4 is finite in FP16; avoid 0 * (-inf) on valid actions.
            logits = mb.select(cond=mb.greater(x=legal_mask, y=np.float32(0)),
                               a=logits, b=np.float32(-1e4), name="legal_logits")
        else:
            penalty = mb.mul(x=mb.sub(x=np.float32(1), y=legal_mask), y=np.float32(-1e9))
            logits = mb.add(x=logits, y=penalty, name="legal_logits")
        value = dense(mb.relu(x=dense(pooled, "value/hidden")), "value/logits")
        value = mb.identity(x=value, name="value_logits")
        policy = mb.softmax(x=logits, axis=-1, name="policy_probabilities")
        wdl = mb.softmax(x=value, axis=-1, name="value_probabilities")
        return (policy, wdl, logits, value) if debug_logits else (policy, wdl)

    if used != set(weights):
        raise ValueError("unused checkpoint parameters: " + str(set(weights) - used))
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--precision", choices=("float32", "float16"), default="float32",
                        help="float16 is an explicit approximate ANE exploration variant")
    parser.add_argument("--debug-logits", action="store_true", help="also export raw policy/value logits")
    parser.add_argument("--inspect-only", action="store_true", help="verify checkpoint without building MIL")
    parser.add_argument("--build-only", action="store_true", help="validate MIL without converting/loading it")
    args = parser.parse_args()
    if args.batch <= 0 or args.capacity <= 0:
        parser.error("batch and capacity must be positive")
    if not (args.inspect_only or args.build_only) and (args.output is None or args.output.suffix != ".mlpackage"):
        parser.error("conversion requires a new output.mlpackage")
    if args.output and (args.output.exists() or args.output.with_suffix(".export.json").exists()):
        parser.error("output already exists")
    metadata, weights, digest = read_checkpoint(args.checkpoint)
    report = {"checkpoint_payload_sha256": digest, "configuration": metadata["configuration"],
              "training_step": metadata["trainingStep"], "batch": args.batch, "capacity": args.capacity,
              "precision": args.precision, "approximate": args.precision == "float16",
              "mask_semantics": "select(legal_mask > 0, logits, -1e4)" if args.precision == "float16"
                  else "logits + (1 - legal_mask) * -1e9",
              "debug_logits": args.debug_logits,
              "numerical_parity": "NOT VALIDATED; no bitwise equivalence claim",
              "ane_activity": "NOT MEASURED", "parameter_count": sum(x.size for x in weights.values())}
    if not args.inspect_only:
        import coremltools as ct
        import numpy as np
        report["coremltools"] = ct.__version__
        program = build_program(metadata["configuration"], weights, args.batch, args.capacity,
                                args.precision, args.debug_logits)
        program.validate()
        report["mil_operations"] = len(program.functions["main"].operations)
        if not args.build_only:
            model = ct.convert(program, source="milinternal", convert_to="mlprogram",
                               minimum_deployment_target=ct.target.macOS13,
                               compute_precision=ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32,
                               outputs=[ct.TensorType(name=x.name, dtype=np.float32)
                                        for x in program.functions["main"].outputs],
                               compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
            model.user_defined_metadata["paisho.checkpoint_sha256"] = digest
            model.user_defined_metadata["paisho.parity"] = report["numerical_parity"]
            model.user_defined_metadata["paisho.precision"] = args.precision
            model.user_defined_metadata["paisho.mask_semantics"] = report["mask_semantics"]
            model.save(str(args.output))
            report["package"] = str(args.output.resolve())
            report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_name("coreml_checkpoint.py"))}
            with args.output.with_suffix(".export.json").open("x") as stream:
                json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
