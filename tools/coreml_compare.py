"""Compare packed JSON reference batches with Core ML. Prediction runs only via CLI.

Optional timings measure synchronous Python predict calls, not isolated device time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

OUTPUTS = ("policy_probabilities", "value_probabilities")
INDEX_LIMITS = {"family_indices": 7, "tile_indices": 12,
                "destination_indices": 289, "pair_indices": 289 * 289}


def input_shapes(batch, capacity):
    return {"spatial": (batch, 17, 17, 29), "global_features": (batch, 26),
            **{name: (batch, capacity) for name in (
                "family_indices", "tile_indices", "tile_presence", "destination_indices",
                "destination_presence", "pair_indices", "pair_presence", "legal_mask")}}


def flat_array(values, shape, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != int(np.prod(shape)) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name}: expected flat numeric array with {int(np.prod(shape))} entries")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name}: nonfinite fixture")
    return raw.reshape(shape)


def prepare_fixture(fixture):
    batch, capacity = fixture["batch"], fixture["capacity"]
    if any(type(x) is not int or x <= 0 for x in (batch, capacity)):
        raise ValueError("batch/capacity must be positive integers")
    if not isinstance(fixture["batches"], list) or not fixture["batches"]:
        raise ValueError("batches must be a nonempty list")
    shapes = input_shapes(batch, capacity)
    prepared = []
    for index, item in enumerate(fixture["batches"]):
        if set(item["inputs"]) != set(shapes):
            raise ValueError(f"batch {index}: input names must be {sorted(shapes)}")
        inputs = {}
        for name, shape in shapes.items():
            raw = flat_array(item["inputs"][name], shape, name)
            if name in INDEX_LIMITS:
                if np.any(raw != np.floor(raw)) or np.any(raw < 0) or np.any(raw >= INDEX_LIMITS[name]):
                    raise ValueError(f"{name}: out-of-range or fractional index (including padding)")
                inputs[name] = raw.astype(np.int32)
            else:
                if np.any(np.abs(raw) > np.finfo(np.float32).max):
                    raise ValueError(f"{name}: float32 overflow")
                if name.endswith("presence") or name == "legal_mask":
                    if not np.isin(raw, [0, 1]).all():
                        raise ValueError(f"{name}: expected binary mask")
                inputs[name] = raw.astype(np.float32)
        if np.any(inputs["legal_mask"].sum(axis=1) == 0):
            raise ValueError("each row needs at least one legal action")
        reference = {}
        for name, width in zip(OUTPUTS, (capacity, 3)):
            raw = flat_array(item[name], (batch, width), name).astype(np.float64)
            if np.any(raw < 0) or np.any(raw > 1) or not np.allclose(raw.sum(axis=1), 1, atol=1e-4, rtol=0):
                raise ValueError(f"{name}: invalid reference probabilities")
            reference[name] = raw
        prepared.append((inputs, reference))
    return prepared


def metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"output shape {actual.shape}, expected {reference.shape}")
    finite = np.isfinite(actual)
    rows = finite.all(axis=1)
    errors = np.abs(actual[finite] - reference[finite])
    changed = np.flatnonzero(rows & (np.argmax(np.where(finite, actual, -np.inf), axis=1)
                                   != np.argmax(reference, axis=1)))
    return {"all_finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "max_abs": float(errors.max()) if errors.size else None,
            "mean_abs": float(errors.mean()) if errors.size else None,
            "error_scope": "finite entries only; nonfinite entries reported separately",
            "argmax_changed_rows": changed.tolist(), "argmax_changed_count": len(changed),
            "argmax_uncomparable_rows": np.flatnonzero(~rows).tolist(),
            "max_probability_sum_error": float(np.max(np.abs(actual[rows].sum(axis=1) - 1))) if rows.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--compute-units", choices=("cpu", "cpu-ane", "cpu-gpu", "all"), default="cpu")
    parser.add_argument("--warmup", type=int, default=0, help="untimed complete fixture sweeps")
    parser.add_argument("--repetitions", type=int, default=0, help="timed complete fixture sweeps; disabled by default")
    parser.add_argument("--max-abs", type=float, help="optional maximum error for either output; no implicit tolerance")
    parser.add_argument("--output", type=Path, help="new JSON report; default stdout only")
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 0:
        parser.error("warmup and repetitions must be nonnegative")
    if args.repetitions and not args.warmup:
        parser.error("timing requires --warmup >= 1")
    if args.max_abs is not None and (not np.isfinite(args.max_abs) or args.max_abs < 0):
        parser.error("max-abs must be finite and nonnegative")
    if args.output and args.output.exists():
        parser.error("report already exists")
    payload = args.fixture.read_bytes()
    fixture = json.loads(payload)
    prepared = prepare_fixture(fixture)  # Validate all data before loading hardware.
    import coremltools as ct
    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu-ane": ct.ComputeUnit.CPU_AND_NE,
             "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL}
    loader = ct.models.CompiledMLModel if args.model.suffix == ".mlmodelc" else ct.models.MLModel
    model = loader(str(args.model), compute_units=units[args.compute_units])
    report = {"model": str(args.model.resolve()), "fixture_sha256": hashlib.sha256(payload).hexdigest(),
              "batch": fixture["batch"], "capacity": fixture["capacity"],
              "compute_units": args.compute_units, "coremltools": ct.__version__,
              "platform": platform.platform(), "max_abs_threshold": args.max_abs,
              "ane_activity": "NOT MEASURED; allowed compute units do not prove placement",
              "batches": []}
    if hasattr(model, "user_defined_metadata"):
        report["model_metadata"] = dict(model.user_defined_metadata)
    for inputs, reference in prepared:
        prediction = model.predict(inputs)
        entry = {name: metrics(prediction[name], reference[name]) for name in OUTPUTS}
        policy = np.asarray(prediction[OUTPUTS[0]])
        entry["policy_padding_mass"] = [float(x) if np.isfinite(x) else None for x in
            np.where(inputs["legal_mask"] > 0, 0, policy).sum(axis=1)]
        report["batches"].append(entry)
    for _ in range(args.warmup):
        for inputs, _ in prepared:
            model.predict(inputs)
    if args.repetitions:
        durations = []
        for _ in range(args.repetitions):
            for inputs, _ in prepared:
                start = time.perf_counter()
                model.predict(inputs)
                durations.append(time.perf_counter() - start)
        report["timing"] = {"scope": "synchronous Python predict including marshaling; excludes load, comparison and warmup",
                            "warmup_sweeps": args.warmup, "repetition_sweeps": args.repetitions,
                            "calls": len(durations), "mean_seconds": float(np.mean(durations)),
                            "median_seconds": float(np.median(durations)), "min_seconds": min(durations),
                            "positions_per_second": fixture["batch"] * len(durations) / sum(durations)}
    results = [row[name] for row in report["batches"] for name in OUTPUTS]
    failed = any(not row["all_finite"] or (args.max_abs is not None and row["max_abs"] > args.max_abs)
                 for row in results)
    report["checks_passed"] = not failed
    report["interpretation"] = "Checks cover finiteness and optional tolerance only; no equivalence claim. Argmax uses first index for ties."
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
