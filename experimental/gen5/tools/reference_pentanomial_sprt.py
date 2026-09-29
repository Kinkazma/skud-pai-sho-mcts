#!/usr/bin/env python3
"""Read-only oracle over a pinned official Fishtest checkout."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys


PINNED_COMMIT = "b8eecff220b562a0dc2c4e68d1fa02521e06d72c"
SOURCE_PATH = pathlib.Path("server/fishtest/stats/LLRcalc.py")
CASES = (
    ((10, 20, 40, 20, 10), 0.0, 5.0),
    ((2, 8, 20, 30, 40), 0.0, 20.0),
    ((0, 0, 3, 5, 12), -5.0, 15.0),
    ((30, 20, 10, 4, 1), 0.0, 10.0),
    ((120, 40, 10, 0, 0), 0.0, 10.0),
    ((0, 0, 10, 40, 120), 0.0, 10.0),
    ((3657864, 21588, 83, 976477, 2), 507.59481790880545, 910.9781759809688),
)


def checked_revision(checkout: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_reference(checkout: pathlib.Path):
    source = checkout / SOURCE_PATH
    specification = importlib.util.spec_from_file_location("pinned_fishtest_llr", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: reference_pentanomial_sprt.py /path/to/fishtest")
    checkout = pathlib.Path(sys.argv[1]).resolve()
    revision = checked_revision(checkout)
    if revision != PINNED_COMMIT:
        raise SystemExit(f"expected Fishtest {PINNED_COMMIT}, got {revision}")
    reference = load_reference(checkout)
    print("PAISHO-PENTANOMIAL-SPRT-OFFICIAL-REFERENCE\t1")
    print(f"source_commit\t{PINNED_COMMIT}")
    print(f"source_path\t{SOURCE_PATH}")
    print("counts\telo0\telo1\tllr")
    for counts, elo0, elo1 in CASES:
        llr = reference.LLR_logistic(elo0, elo1, list(counts))
        fields = ",".join(str(count) for count in counts)
        print(f"{fields}\t{elo0:.17g}\t{elo1:.17g}\t{llr:.15f}")


if __name__ == "__main__":
    main()
