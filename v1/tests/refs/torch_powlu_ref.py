#!/usr/bin/env python3
# /// script
# dependencies = ["torch>=2.3", "numpy>=1.26"]
# ///
"""PyTorch reference for gradients.c powlu forward + backward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def powlu(x1: torch.Tensor, x2: torch.Tensor, m: float) -> torch.Tensor:
    z = x2.float()
    s = torch.sigmoid(z)
    pos_mask = z > 0.0
    # torch.where still builds both branches for autograd. Keep the inactive
    # positive branch finite at z<=0, or branch-boundary tests produce NaN grads.
    zsafe = torch.where(pos_mask, z, torch.ones_like(z))
    neg = z * s
    r = torch.sqrt(zsafe)
    a = m / (r + 1.0)
    pos = torch.pow(zsafe, a) * s
    gate = torch.where(pos_mask, pos, neg)
    return (x1.float() * gate).to(x1.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--m", type=float, default=3.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    x1 = (torch.randn(args.n, device=dev, dtype=dtype) * 0.7).requires_grad_()
    # Include tiny positives near branch boundary; common NaN source for analytic bwd.
    x2_data = torch.randn(args.n, device=dev, dtype=dtype) * 0.7
    if args.n >= 8:
        tiny = torch.tensor([0.0, 2.0**-24, 2.0**-20, 2.0**-14,
                             -2.0**-24, -2.0**-20, -2.0**-14, 1.0],
                            device=dev, dtype=dtype)
        x2_data[:8] = tiny
    x2 = x2_data.requires_grad_()
    go = torch.randn(args.n, device=dev, dtype=dtype) * 0.03

    out = powlu(x1, x2, args.m)
    (out.float() * go.float()).sum().backward()

    ref = {
        "x1": x1.detach().cpu().numpy(),
        "x2": x2.detach().cpu().numpy(),
        "go": go.detach().cpu().numpy(),
        "out": out.detach().cpu().numpy(),
        "grad_x1": x1.grad.detach().cpu().numpy(),
        "grad_x2": x2.grad.detach().cpu().numpy(),
    }
    if args.out:
        meta = dict(op="powlu", dtype=args.dtype, device=str(dev), n=args.n, m=args.m)
        np.savez(args.out, **ref, meta=np.array(json.dumps(meta)))
        print(f"wrote {args.out}")
    if args.candidate:
        return compare(ref, args.candidate, args.atol, args.rtol)
    for key in ["out", "grad_x1", "grad_x2"]:
        a = ref[key].astype(np.float64)
        print(f"{key}: finite={np.isfinite(a).all()} min={np.nanmin(a):.6g} max={np.nanmax(a):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
