"""Lightweight binary reader + MIL shape tests. No conversion or prediction."""
import hashlib
import json
import math
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

from coreml_export import build_program
from coreml_checkpoint import expected_shapes, read_checkpoint
import numpy as np


CONFIG = dict(trunkChannels=2, residualBlocks=1, policyEmbeddingChannels=2,
              valueHiddenChannels=2, normalizationEpsilon=1e-5)


def fixture():
    metadata = dict(formatVersion=2, tensorSchema="paisho-neural-encoding-v1",
                    ruleProfile="skud-pai-sho-2022-03-14", configuration=CONFIG,
                    trainingStep=5, progress={"scheduler": {"completedSteps": 5}})
    encoded = json.dumps(metadata).encode()
    shapes = expected_shapes(CONFIG)
    body = b"PAISHO-CKPT-V2\n" + struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(shapes))
    for name, shape in shapes.items():
        name_bytes = name.encode()
        count = math.prod(shape)
        values = np.linspace(-0.25, 0.25, count, dtype=np.float32)
        body += struct.pack("<I", len(name_bytes)) + name_bytes + struct.pack("<I", len(shape))
        body += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<Q", count)
        body += values.tobytes() + bytes(count * 8)
    return body + hashlib.sha256(body).digest()


class CoreMLStudyTests(unittest.TestCase):
    def read(self, data):
        # Avoid creating test files outside the task's permitted paths.
        with patch.object(Path, "read_bytes", return_value=data):
            return read_checkpoint("fixture")

    def test_values_and_checksum(self):
        data = fixture()
        meta, weights, digest = self.read(data)
        self.assertEqual(meta["trainingStep"], 5)
        self.assertEqual(digest, data[-32:].hex())
        np.testing.assert_array_equal(weights["stem/global/weights"].ravel(),
                                     np.linspace(-0.25, 0.25, 52, dtype=np.float32))

    def test_corruption_and_truncation(self):
        data = bytearray(fixture())
        data[-50] ^= 1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.read(data)
        with self.assertRaises(ValueError):
            self.read(b"short")

    def test_trailing_bytes_with_valid_digest(self):
        body = fixture()[:-32] + b"extra"
        with self.assertRaisesRegex(ValueError, "trailing"):
            self.read(body + hashlib.sha256(body).digest())

    def test_wrong_shape_with_valid_digest(self):
        body = fixture()[:-32]
        old = b"stem/conv/weights" + struct.pack("<IQ", 4, 3)
        new = b"stem/conv/weights" + struct.pack("<IQ", 4, 4)
        self.assertIn(old, body)
        body = body.replace(old, new, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.read(body + hashlib.sha256(body).digest())

    def test_mil_shapes_and_fp32(self):
        import coremltools as ct
        meta, weights, _ = self.read(fixture())
        # Guard against accidentally adding conversion to this lightweight test.
        with patch.object(ct, "convert", side_effect=AssertionError("conversion prohibited")):
            program = build_program(meta["configuration"], weights, batch=2, capacity=5)
        program.validate()
        outputs = program.functions["main"].outputs
        self.assertEqual([x.shape for x in outputs], [(2, 5), (2, 3)])
        self.assertEqual([x.name for x in outputs],
                         ["policy_probabilities", "value_probabilities"])
        self.assertTrue(all(x.dtype.__name__ == "fp32" for x in outputs))

    def test_fp16_variant_mask_and_debug_outputs(self):
        meta, weights, _ = self.read(fixture())
        program = build_program(meta["configuration"], weights, 2, 5,
                                precision="float16", debug_logits=True)
        program.validate()
        function = program.functions["main"]
        self.assertEqual([x.name for x in function.outputs],
                         ["policy_probabilities", "value_probabilities", "legal_logits", "value_logits"])
        mask = next(op for op in function.operations if op.name == "legal_logits")
        self.assertEqual(mask.op_type, "select")
        self.assertEqual(float(mask.b.val), -1e4)
        self.assertTrue(np.isfinite(np.float16(mask.b.val)))

    def test_comparison_nonfinite_and_argmax(self):
        from coreml_compare import metrics
        reference = np.array([[0.7, 0.3], [0.4, 0.6]])
        result = metrics([[0.2, 0.8], [float("nan"), 0.6]], reference)
        self.assertFalse(result["all_finite"])
        self.assertEqual(result["nonfinite_count"], 1)
        self.assertEqual(result["argmax_changed_rows"], [0])
        self.assertEqual(result["argmax_uncomparable_rows"], [1])
        self.assertAlmostEqual(result["max_abs"], 0.5)

    def test_fixture_validation(self):
        from coreml_compare import input_shapes, prepare_fixture
        inputs = {name: [0] * math.prod(shape) for name, shape in input_shapes(1, 2).items()}
        inputs["legal_mask"] = [1, 0]
        document = {"batch": 1, "capacity": 2, "batches": [{"inputs": inputs,
                    "policy_probabilities": [1, 0], "value_probabilities": [0.5, 0.25, 0.25]}]}
        prepared = prepare_fixture(document)
        self.assertEqual(prepared[0][0]["spatial"].shape, (1, 17, 17, 29))
        self.assertEqual(prepared[0][0]["family_indices"].dtype, np.int32)
        inputs["family_indices"] = [0, 0.5]
        with self.assertRaisesRegex(ValueError, "fractional"):
            prepare_fixture(document)


if __name__ == "__main__":
    unittest.main()
