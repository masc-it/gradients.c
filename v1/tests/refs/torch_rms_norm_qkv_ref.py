#!/usr/bin/env python3
# /// script
# dependencies = ["torch>=2.3", "numpy>=1.26"]
# ///
"""PyTorch reference for gradients.c rms_norm_qkv forward + backward.

Usage:
  uv run tests/refs/torch_rms_norm_qkv_ref.py --out /tmp/rnqkv_ref.npz
  uv run tests/refs/torch_rms_norm_qkv_ref.py --candidate actual.npz

Candidate NPZ should use same output/grad keys printed by --out.
"""

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


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return float(np.max(np.abs(a - b) / denom)) if a.size else 0.0


def compare(ref: dict[str, np.ndarray], candidate: Path, atol: float, rtol: float) -> int:
    got = np.load(candidate)
    failures = 0
    for key, exp in ref.items():
        if key == "meta":
            continue
        if key not in got:
            print(f"MISSING {key}")
            failures += 1
            continue
        act = got[key]
        abs_max = float(np.max(np.abs(act - exp))) if exp.size else 0.0
        rel_max = rel_err(act, exp)
        ok = abs_max <= atol + rtol * float(np.max(np.abs(exp)) if exp.size else 0.0)
        print(f"{key}: abs={abs_max:.6g} rel={rel_max:.6g} {'OK' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    return failures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--rows", type=int, default=17)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--q", type=int, default=32)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--v", type=int, default=32)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--out", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--atol", type=float, default=3e-2)
    p.add_argument("--rtol", type=float, default=3e-2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    dtype = getattr(torch, args.dtype)

    # Keep magnitudes GPT-like. Inputs are leaf half tensors; reductions/project
    # math is expressed in fp32 where gradients.c CPU/MPS paths accumulate fp32.
    x = (torch.randn(args.rows, args.d, device=dev, dtype=dtype) * 0.7).requires_grad_()
    gamma = (1.0 + 0.01 * torch.randn(args.d, device=dev, dtype=dtype)).requires_grad_()
    wq = (torch.randn(args.d, args.q, device=dev, dtype=dtype) * 0.08).requires_grad_()
    wk = (torch.randn(args.d, args.k, device=dev, dtype=dtype) * 0.08).requires_grad_()
    wv = (torch.randn(args.d, args.v, device=dev, dtype=dtype) * 0.08).requires_grad_()

    inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + args.eps)
    # Metal path materializes norm as dtype before Q/K/V MPS GEMMs.
    norm = (x.float() * inv * gamma.float()).to(dtype)
    q = (norm.float() @ wq.float()).to(dtype)
    k = (norm.float() @ wk.float()).to(dtype)
    v = (norm.float() @ wv.float()).to(dtype)

    go_norm = torch.randn_like(norm) * 0.02
    go_q = torch.randn_like(q) * 0.02
    go_k = torch.randn_like(k) * 0.02
    go_v = torch.randn_like(v) * 0.02
    loss = (norm.float() * go_norm.float()).sum()
    loss = loss + (q.float() * go_q.float()).sum()
    loss = loss + (k.float() * go_k.float()).sum()
    loss = loss + (v.float() * go_v.float()).sum()
    loss.backward()

    ref = {
        "x": x.detach().cpu().numpy(),
        "gamma": gamma.detach().cpu().numpy(),
        "wq": wq.detach().cpu().numpy(),
        "wk": wk.detach().cpu().numpy(),
        "wv": wv.detach().cpu().numpy(),
        "go_norm": go_norm.detach().cpu().numpy(),
        "go_q": go_q.detach().cpu().numpy(),
        "go_k": go_k.detach().cpu().numpy(),
        "go_v": go_v.detach().cpu().numpy(),
        "out_norm": norm.detach().cpu().numpy(),
        "out_q": q.detach().cpu().numpy(),
        "out_k": k.detach().cpu().numpy(),
        "out_v": v.detach().cpu().numpy(),
        "grad_x": x.grad.detach().cpu().numpy(),
        "grad_gamma": gamma.grad.detach().cpu().numpy(),
        "grad_wq": wq.grad.detach().cpu().numpy(),
        "grad_wk": wk.grad.detach().cpu().numpy(),
        "grad_wv": wv.grad.detach().cpu().numpy(),
    }
    meta = dict(op="rms_norm_qkv", dtype=args.dtype, device=str(dev), eps=args.eps,
                rows=args.rows, d=args.d, q=args.q, k=args.k, v=args.v)
    if args.out:
        np.savez(args.out, **ref, meta=np.array(json.dumps(meta)))
        print(f"wrote {args.out}")
    if args.candidate:
        return compare(ref, args.candidate, args.atol, args.rtol)
    for key in ["out_norm", "out_q", "out_k", "out_v", "grad_x", "grad_gamma"]:
        a = ref[key].astype(np.float64)
        print(f"{key}: min={a.min():.6g} max={a.max():.6g} mean={a.mean():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
