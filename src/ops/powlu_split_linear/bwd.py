# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Backward/autograd correctness harness template for gd_powlu_split_linear.

Fill in a C runner that records gd_powlu_split_linear on the autograd tape, calls
gd_backward or gd_backward_many, then compare gradients against
PyTorch autograd. Run from the repository root with:

    uv run src/ops/powlu_split_linear/bwd.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def main() -> None:
    root = repo_root()
    build_library(root)
    _ = (np, torch)
    print("TODO: implement gd_powlu_split_linear backward PyTorch/autograd comparison")


if __name__ == "__main__":
    main()
