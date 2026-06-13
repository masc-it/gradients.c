# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c rms_norm forward correctness check against PyTorch formula."""

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
    float *y_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, y;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 7 || parse_dtype(argv[2], &dtype) != 0 || parse_f32(argv[3], &eps) != 0 ||
        parse_u32(argv[4], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        argc != (int)(7U + rank)) {
        fprintf(stderr, "usage: %s OUT.bin f16|f32 EPS RANK DIMS... X.f32 W.f32\n", argv[0]);
        return 2;
    }
    arg = 5;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &shape[d]) != 0 || shape[d] <= 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)shape[d];
    }
    cols = (size_t)shape[rank - 1U];
    w_shape[0] = (int64_t)cols;
    x_host = (float *)malloc(sizeof(float) * count);
    w_host = (float *)malloc(sizeof(float) * cols);
    y_host = (float *)malloc(sizeof(float) * count);
    if (x_host == NULL || w_host == NULL || y_host == NULL ||
        read_file(argv[arg], x_host, sizeof(float) * count) != 0 ||
        read_file(argv[arg + 1], w_host, sizeof(float) * cols) != 0) { goto fail; }

    oracle_memory_config_slots(&cfg,
                               align_up(sizeof(float) * (count + cols) * 4U + 1024U * 1024U, 4096U),
                               align_up(sizeof(float) * count * 12U + 1024U * 1024U, 4096U),
                               3U,
                               2U);
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), x_host, count, false, &x));
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(1U, w_shape), w_host, cols, false, &w));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK(ctx, gd_rms_norm(ctx, &x, &w, eps, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &y, y_host, count));
    if (write_file(argv[1], y_host, sizeof(float) * count) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_host); free(w_host); free(y_host);
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
    profile = os.environ.get("GD_RMS_NORM_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RMS_NORM_FWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "f32": torch.float32}[dtype]


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x524D534E) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.5
    w = torch.randn((case.shape[-1],), generator=gen, dtype=torch.float32) * 0.25 + 1.0
    return x, w


def reference(case: Case, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    dtype = torch_dtype(case.dtype)
    xq = x.to(dtype)
    wq = w.to(dtype)
    inv = torch.rsqrt(xq.float().pow(2).mean(dim=-1, keepdim=True) + case.eps)
    y = xq.float() * wq.float() * inv
    return y.to(dtype).float() if dtype == torch.float16 else y.float()


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x, w = make_inputs(case)
    want = reference(case, x, w).contiguous()
    x_path = tmp / f"{case.name}_x.f32"
    w_path = tmp / f"{case.name}_w.f32"
    y_path = tmp / f"{case.name}_y.f32"
    x_path.write_bytes(x.contiguous().numpy().astype(np.float32).tobytes())
    w_path.write_bytes(w.contiguous().numpy().astype(np.float32).tobytes())
    cmd = [str(binary), str(y_path), case.dtype, str(case.eps), str(len(case.shape)),
           *map(str, case.shape), str(x_path), str(w_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(y_path.read_bytes(), dtype=np.float32).reshape(case.shape)
    atol = 2.5e-3 if case.dtype == "f16" else 2.0e-5
    rtol = 2.5e-3 if case.dtype == "f16" else 2.0e-5
    np.testing.assert_allclose(got, want.numpy(), atol=atol, rtol=rtol)
    print(f"[rms_norm/fwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_rms_norm_fwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
