#!/usr/bin/env python3
# /// script
# dependencies = ["torch>=2.3", "numpy>=1.26"]
# ///
"""PyTorch reference for fused lm_cross_entropy forward + backward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compare(ref: dict[str, np.ndarray], candidate: Path, atol: float, rtol: float) -> int:
    got = np.load(candidate)
    bad = 0
    for key, exp in ref.items():
        if key == "meta":
            continue
        if key not in got:
            print(f"MISSING {key}")
            bad += 1
            continue
        act = got[key]
        abs_max = float(np.max(np.abs(act - exp))) if exp.size else 0.0
        denom = np.maximum(np.maximum(np.abs(act), np.abs(exp)), 1e-12)
        rel_max = float(np.max(np.abs(act - exp) / denom)) if exp.size else 0.0
        ok = abs_max <= atol + rtol * float(np.max(np.abs(exp)) if exp.size else 0.0)
        print(f"{key}: abs={abs_max:.6g} rel={rel_max:.6g} {'OK' if ok else 'FAIL'}")
        bad += 0 if ok else 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rows", type=int, default=257)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=211)
    ap.add_argument("--ignore-index", type=int, default=-100)
    ap.add_argument("--ignore-frac", type=float, default=0.35)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--atol", type=float, default=3e-2)
    ap.add_argument("--rtol", type=float, default=3e-2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    hidden = (torch.randn(args.rows, args.d, device=dev, dtype=dtype) * 0.2).requires_grad_()
    weight = (torch.randn(args.vocab, args.d, device=dev, dtype=dtype) * 0.08).requires_grad_()
    targets = torch.randint(0, args.vocab, (args.rows,), device=dev, dtype=torch.long)
    ignore = torch.rand(args.rows, device=dev) < args.ignore_frac
    targets[ignore] = args.ignore_index

    logits = hidden.float() @ weight.float().t()
    loss = F.cross_entropy(logits, targets, ignore_index=args.ignore_index, reduction="mean")
    loss.backward()

    # Match gradients.c aux outputs: row max m and exp-sum l for every row.
    m = logits.max(dim=-1).values
    l = torch.exp(logits - m[:, None]).sum(dim=-1)

    ref = {
        "hidden": hidden.detach().cpu().numpy(),
        "weight": weight.detach().cpu().numpy(),
        "targets": targets.detach().cpu().numpy().astype(np.int32),
        "loss": np.array(loss.detach().cpu().item(), dtype=np.float32),
        "m": m.detach().cpu().numpy().astype(np.float32),
        "l": l.detach().cpu().numpy().astype(np.float32),
        "grad_hidden": hidden.grad.detach().cpu().numpy(),
        "grad_weight": weight.grad.detach().cpu().numpy(),
    }
    if args.out:
        meta = dict(op="lm_cross_entropy", dtype=args.dtype, device=str(dev), rows=args.rows,
                    d=args.d, vocab=args.vocab, ignore_index=args.ignore_index)
        np.savez(args.out, **ref, meta=np.array(json.dumps(meta)))
        print(f"wrote {args.out}")
    if args.candidate:
        return compare(ref, args.candidate, args.atol, args.rtol)
    for key in ["loss", "m", "l", "grad_hidden", "grad_weight"]:
        a = ref[key].astype(np.float64)
        print(f"{key}: finite={np.isfinite(a).all()} min={np.nanmin(a):.6g} max={np.nanmax(a):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
