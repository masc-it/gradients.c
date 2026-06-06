#!/usr/bin/env python3
"""Prepare the XOR example as a tiny GDDS dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gdds_utils import FieldSpec, tensor, write_gdds_split  # noqa: E402


FIELDS = [
    FieldSpec("x", "f16", (2,), collate="stack"),
    FieldSpec("target", "f16", (1,), collate="stack"),
]

SAMPLES = [
    {"x": tensor([0.0, 0.0], "f16"), "target": tensor([0.0], "f16")},
    {"x": tensor([0.0, 1.0], "f16"), "target": tensor([1.0], "f16")},
    {"x": tensor([1.0, 0.0], "f16"), "target": tensor([1.0], "f16")},
    {"x": tensor([1.0, 1.0], "f16"), "target": tensor([0.0], "f16")},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--split", default="xor")
    args = parser.parse_args()

    paths = write_gdds_split(
        args.out_dir,
        args.split,
        FIELDS,
        SAMPLES,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
