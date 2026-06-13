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

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.op_oracle import build_library

import numpy as np
import torch


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    _ = (np, torch)
    print("TODO: implement gd_powlu_split_linear backward PyTorch/autograd comparison")


if __name__ == "__main__":
    main()
