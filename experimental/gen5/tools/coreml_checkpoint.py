"""Read inference weights from the existing checksummed PAISHO-CKPT-V2 format."""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def expected_shapes(config):
    c, r, e, h = (config[k] for k in (
        "trunkChannels", "residualBlocks", "policyEmbeddingChannels", "valueHiddenChannels"))
    if any(type(v) is not int or v <= 0 for v in (c, r, e, h)):
        raise ValueError("invalid network configuration")
    eps = config["normalizationEpsilon"]
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid normalization epsilon")
    shapes = {}

    def affine(name, weight):
        shapes[name + "/weights"] = weight
        shapes[name + "/bias"] = (weight[-1],)

    def norm(name):
        for suffix in ("gamma", "beta"):
            shapes[name + "/" + suffix] = (1, 1, 1, c)

    affine("stem/conv", (3, 3, 29, c))
    affine("stem/global", (26, c))
    norm("stem/norm")
    for block in range(r):
        for layer in (1, 2):
            affine(f"trunk/block_{block}/conv_{layer}", (3, 3, c, c))
            norm(f"trunk/block_{block}/norm_{layer}")
    for name, count in (("family", 7), ("tile", 12)):
        affine("policy/" + name, (c, count))
    for name, count in (("destination", 1), ("source_embedding", e), ("destination_embedding", e)):
        affine("policy/" + name, (1, 1, c, count))
    affine("value/hidden", (c, h))
    affine("value/logits", (h, 3))
    return shapes


def read_checkpoint(path):
    data = Path(path).read_bytes()
    body = memoryview(data)[:-32]
    if len(data) < 46 or hashlib.sha256(body).digest() != data[-32:]:
        raise ValueError("checkpoint SHA-256 mismatch or truncated file")
    offset = 0

    def take(n):
        nonlocal offset
        if n < 0 or n > len(body) - offset:
            raise ValueError("truncated checkpoint")
        value = body[offset:offset + n]
        offset += n
        return value

    def integer(code):
        return struct.unpack("<" + code, take(struct.calcsize(code)))[0]

    magic = b"PAISHO-CKPT-V2\n"
    if bytes(take(len(magic))) != magic:
        raise ValueError("unsupported checkpoint magic")
    metadata_size = integer("I")
    if metadata_size > 16 * 1024 * 1024:
        raise ValueError("oversized metadata")
    metadata = json.loads(bytes(take(metadata_size)))
    for key, expected in (("formatVersion", 2), ("tensorSchema", "paisho-neural-encoding-v1"),
                          ("ruleProfile", "skud-pai-sho-2022-03-14")):
        if metadata.get(key) != expected:
            raise ValueError(f"unsupported {key}")
    if metadata["trainingStep"] != metadata["progress"]["scheduler"]["completedSteps"]:
        raise ValueError("scheduler step mismatch")
    shapes = expected_shapes(metadata["configuration"])
    count = integer("I")
    if count != len(shapes):
        raise ValueError("unexpected parameter count")
    parameters = {}
    for _ in range(count):
        length = integer("I")
        if not 0 < length <= 4096:
            raise ValueError("invalid parameter name length")
        name = bytes(take(length)).decode("utf-8")
        rank = integer("I")
        if not 0 < rank <= 16:
            raise ValueError("invalid rank")
        shape = tuple(integer("Q") for _ in range(rank))
        size = integer("Q")
        if name in parameters or shapes.get(name) != shape or size != math.prod(shape):
            raise ValueError(f"unexpected/duplicate parameter or shape: {name}")
        for slot in ("values", "momentum", "velocity"):
            array = np.frombuffer(take(4 * size), dtype="<f4")
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite {name}/{slot}")
            if slot == "values":
                parameters[name] = array.reshape(shape).copy()
    if offset != len(body):
        raise ValueError("trailing checkpoint data")
    return metadata, parameters, data[-32:].hex()
