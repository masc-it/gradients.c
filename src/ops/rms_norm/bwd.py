# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c rms_norm backward correctness check against PyTorch formula."""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.op_oracle import build_library, compile_runner, gradients_env

import numpy as np
import torch


@dataclass(frozen=True)
class Case:
    name: str
    dtype: str
    shape: tuple[int, ...]
    eps: float = 1.0e-5


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS] = {0};
    int64_t w_shape[1];
    float eps;
    size_t count = 1U;
    size_t cols;
    float *x_host = NULL;
    float *w_host = NULL;
    float *g_host = NULL;
    float *dx_host = NULL;
    float *dw_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, g, dx, dw;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 9 || parse_dtype(argv[3], &dtype) != 0 || parse_f32(argv[4], &eps) != 0 ||
        parse_u32(argv[5], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        argc != (int)(9U + rank)) {
        fprintf(stderr, "usage: %s DX.f32 DW.f32 f16|f32 EPS RANK DIMS... X.f32 W.f32 G.f32\n", argv[0]);
        return 2;
    }
    arg = 6;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &shape[d]) != 0 || shape[d] <= 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)shape[d];
    }
    cols = (size_t)shape[rank - 1U];
    w_shape[0] = (int64_t)cols;
    x_host = (float *)malloc(sizeof(float) * count);
    w_host = (float *)malloc(sizeof(float) * cols);
    g_host = (float *)malloc(sizeof(float) * count);
    dx_host = (float *)malloc(sizeof(float) * count);
    dw_host = (float *)malloc(sizeof(float) * cols);
    if (x_host == NULL || w_host == NULL || g_host == NULL || dx_host == NULL || dw_host == NULL ||
        read_file(argv[arg], x_host, sizeof(float) * count) != 0 ||
        read_file(argv[arg + 1], w_host, sizeof(float) * cols) != 0 ||
        read_file(argv[arg + 2], g_host, sizeof(float) * count) != 0) { goto fail; }

    oracle_memory_config_slots(&cfg,
                               align_up(sizeof(float) * (count * 2U + cols) * 4U + 1024U * 1024U, 4096U),
                               align_up(sizeof(float) * count * 24U + sizeof(float) * cols * 16U + 1024U * 1024U, 4096U),
                               3U,
                               2U);
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), x_host, count, false, &x));
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(1U, w_shape), w_host, cols, false, &w));
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), g_host, count, false, &g));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_rms_norm_backward(ctx, &x, &w, &g, eps, &dx, &dw));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dx, dx_host, count));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dw, dw_host, cols));
    if (write_file(argv[1], dx_host, sizeof(float) * count) != 0 ||
        write_file(argv[2], dw_host, sizeof(float) * cols) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_host); free(w_host); free(g_host); free(dx_host); free(dw_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("tiny_f32", "f32", (2, 4)),
        Case("tokens_f16", "f16", (4, 17, 64)),
        Case("llama_hidden_f16", "f16", (2, 128, 4096)),
        Case("mlp_f32", "f32", (257, 1024)),
    ]
    profile = os.environ.get("GD_RMS_NORM_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RMS_NORM_BWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "f32": torch.float32}[dtype]


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x42574452) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.5
    w = torch.randn((case.shape[-1],), generator=gen, dtype=torch.float32) * 0.25 + 1.0
    g = torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.25
    return x, w, g


def reference(case: Case, x: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch_dtype(case.dtype)
    xq = x.to(dtype).float()
    wq = w.to(dtype).float()
    gq = g.to(dtype).float()
    cols = xq.shape[-1]
    inv = torch.rsqrt(xq.pow(2).mean(dim=-1, keepdim=True) + case.eps)
    a = (gq * wq * xq).sum(dim=-1, keepdim=True)
    dx = inv * gq * wq - xq * (inv ** 3) * a / float(cols)
    reduce_dims = tuple(range(xq.ndim - 1))
    dw = (gq * xq * inv).sum(dim=reduce_dims)
    if dtype == torch.float16:
        dx = dx.to(torch.float16).float()
        dw = dw.to(torch.float16).float()
    return dx.contiguous(), dw.contiguous()


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x, w, g = make_inputs(case)
    want_dx, want_dw = reference(case, x, w, g)
    x_path = tmp / f"{case.name}_x.f32"
    w_path = tmp / f"{case.name}_w.f32"
    g_path = tmp / f"{case.name}_g.f32"
    dx_path = tmp / f"{case.name}_dx.f32"
    dw_path = tmp / f"{case.name}_dw.f32"
    x_path.write_bytes(x.contiguous().numpy().astype(np.float32).tobytes())
    w_path.write_bytes(w.contiguous().numpy().astype(np.float32).tobytes())
    g_path.write_bytes(g.contiguous().numpy().astype(np.float32).tobytes())
    cmd = [str(binary), str(dx_path), str(dw_path), case.dtype, str(case.eps), str(len(case.shape)),
           *map(str, case.shape), str(x_path), str(w_path), str(g_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got_dx = np.frombuffer(dx_path.read_bytes(), dtype=np.float32).reshape(case.shape)
    got_dw = np.frombuffer(dw_path.read_bytes(), dtype=np.float32).reshape((case.shape[-1],))
    atol = 5.0e-3 if case.dtype == "f16" else 3.0e-5
    rtol = 5.0e-3 if case.dtype == "f16" else 3.0e-5
    np.testing.assert_allclose(got_dx, want_dx.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(got_dw, want_dw.numpy(), atol=atol, rtol=rtol)
    print(f"[rms_norm/bwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_rms_norm_bwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
