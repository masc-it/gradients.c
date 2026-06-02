#!/usr/bin/env python3
# /// script
# dependencies = ["torch>=2.3", "numpy>=1.26"]
# ///
"""PyTorch reference for gradients.c sdpa_varlen forward + backward."""

from __future__ import annotations

import argparse
import json
import math
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


def allowed(i: torch.Tensor, j: torch.Tensor, causal: bool, window: int, prefix_len: int) -> torch.Tensor:
    ok = torch.ones((i.numel(), j.numel()), device=i.device, dtype=torch.bool)
    ii = i[:, None]
    jj = j[None, :]
    if causal:
        if prefix_len > 0:
            ok &= torch.where(ii < prefix_len, jj < prefix_len, jj <= ii)
        else:
            ok &= jj <= ii
    if window > 0:
        if prefix_len > 0:
            ok &= torch.where((ii >= prefix_len) & (jj >= prefix_len), (ii - jj) < window, True)
        else:
            ok &= (ii - jj) < window
    return ok


def sdpa_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor,
                scale: float, causal: bool, window: int, prefix_len: int) -> torch.Tensor:
    outs = []
    hq = q.shape[1]
    hkv = k.shape[1]
    group = hq // hkv
    for b in range(cu.numel() - 1):
        lo = int(cu[b].item())
        hi = int(cu[b + 1].item())
        tq = hi - lo
        pos = torch.arange(tq, device=q.device)
        mask = allowed(pos, pos, causal, window, prefix_len)
        for h in range(hq):
            hk = h // group
            scores = (q[lo:hi, h].float() @ k[lo:hi, hk].float().t()) * scale
            scores = scores.masked_fill(~mask, -torch.inf)
            probs = torch.softmax(scores, dim=-1)
            outs.append((probs @ v[lo:hi, hk].float()).to(q.dtype))
    # outs were [B,H,T,D]; restore packed [N,H,D].
    out = torch.empty_like(q)
    idx = 0
    for b in range(cu.numel() - 1):
        lo = int(cu[b].item())
        hi = int(cu[b + 1].item())
        for h in range(hq):
            out[lo:hi, h] = outs[idx]
            idx += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lengths", default="7,11,13")
    ap.add_argument("--hq", type=int, default=4)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--prefix-len", type=int, default=3)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--atol", type=float, default=4e-2)
    ap.add_argument("--rtol", type=float, default=4e-2)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    total = sum(lengths)
    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(args.dh)
    torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    dtype = getattr(torch, args.dtype)

    q = (torch.randn(total, args.hq, args.dh, device=dev, dtype=dtype) * 0.25).requires_grad_()
    k = (torch.randn(total, args.hkv, args.dh, device=dev, dtype=dtype) * 0.25).requires_grad_()
    v = (torch.randn(total, args.hkv, args.dh, device=dev, dtype=dtype) * 0.25).requires_grad_()
    cu = torch.tensor([0] + list(np.cumsum(lengths)), device=dev, dtype=torch.int32)
    go = torch.randn_like(q) * 0.02

    out = sdpa_varlen(q, k, v, cu, scale, args.causal, args.window, args.prefix_len)
    (out.float() * go.float()).sum().backward()

    ref = {
        "q": q.detach().cpu().numpy(),
        "k": k.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
        "cu": cu.detach().cpu().numpy(),
        "go": go.detach().cpu().numpy(),
        "out": out.detach().cpu().numpy(),
        "grad_q": q.grad.detach().cpu().numpy(),
        "grad_k": k.grad.detach().cpu().numpy(),
        "grad_v": v.grad.detach().cpu().numpy(),
    }
    if args.out:
        meta = dict(op="sdpa_varlen", dtype=args.dtype, device=str(dev), lengths=lengths,
                    hq=args.hq, hkv=args.hkv, dh=args.dh, causal=args.causal,
                    window=args.window, prefix_len=args.prefix_len, scale=scale)
        np.savez(args.out, **ref, meta=np.array(json.dumps(meta)))
        print(f"wrote {args.out}")
    if args.candidate:
        return compare(ref, args.candidate, args.atol, args.rtol)
    for key in ["out", "grad_q", "grad_k", "grad_v"]:
        a = ref[key].astype(np.float64)
        print(f"{key}: finite={np.isfinite(a).all()} min={np.nanmin(a):.6g} max={np.nanmax(a):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
