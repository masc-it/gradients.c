# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c dropout backward correctness check against a PyTorch tensor formula."""

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
    p: float
    seed: int


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    size_t elem_size;
    size_t elems = 1U;
    size_t bytes;
    unsigned char *x_data = NULL;
    unsigned char *g_data = NULL;
    unsigned char *dx_direct_data = NULL;
    unsigned char *dx_auto_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, g, y, dx_direct, dx_auto;
    gd_status st;
    float p;
    uint64_t seed;
    uint32_t i;
    int rc = 1;

    if (argc < 9 || parse_dtype_with_size(argv[5], &dtype, &elem_size) != 0 ||
        parse_float(argv[6], &p) != 0 || parse_u64(argv[7], &seed) != 0 ||
        parse_u32(argv[8], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        argc != (int)(9U + rank)) {
        fprintf(stderr, "usage: %s X.bin G.bin DX_DIRECT.bin DX_AUTO.bin f16|f32 P SEED RANK DIM...\n", argv[0]);
        return 2;
    }
    for (i = 0U; i < rank; ++i) {
        if (parse_i64_dim(argv[9U + i], &shape[i]) != 0 ||
            (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / elems)) {
            return 2;
        }
        elems *= (size_t)shape[i];
    }
    if (elems == 0U || elems > SIZE_MAX / elem_size) { return 2; }
    bytes = elems * elem_size;
    x_data = (unsigned char *)malloc(bytes);
    g_data = (unsigned char *)malloc(bytes);
    dx_direct_data = (unsigned char *)calloc(elems, elem_size);
    dx_auto_data = (unsigned char *)calloc(elems, elem_size);
    if (x_data == NULL || g_data == NULL || dx_direct_data == NULL || dx_auto_data == NULL ||
        read_file(argv[1], x_data, bytes) != 0 || read_file(argv[2], g_data, bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(bytes * 2U + 1024U * 1024U, 4096U),
                         align_up(bytes * 8U + elems * 2U + 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &g));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &g, g_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));

    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_dropout_backward(ctx, &x, &g, p, seed, &dx_direct));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx_direct, dx_direct_data, bytes));

    x.requires_grad = true;
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_dropout(ctx, &x, p, true, seed, &y));
    CHECK(ctx, gd_backward(ctx, &y, &g));
    CHECK(ctx, gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx_auto, dx_auto_data, bytes));

    if (write_file(argv[3], dx_direct_data, bytes) != 0 || write_file(argv[4], dx_auto_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(g_data); free(dx_direct_data); free(dx_auto_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    smoke = [
        Case("tail_1x513_f16", "f16", (1, 513), 0.20, 0x1234_5678_9ABC_DEF0),
        Case("mlp_hidden_128x128_f16", "f16", (128, 128), 0.10, 0x0DDC_0FFEE_0001),
        Case("small_rank3_f32", "f32", (3, 17, 65), 0.35, 0xCAFE_BABE_1234),
    ]
    classic = [
        Case("act_4096x1024_f16", "f16", (4096, 1024), 0.10, 0xABCDEF01),
        Case("rank3_8x512x1024_f16", "f16", (8, 512, 1024), 0.20, 0xABCDEF02),
        Case("act_2048x2048_f32", "f32", (2048, 2048), 0.10, 0xABCDEF03),
    ]
    profile = os.environ.get("GD_DROPOUT_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_DROPOUT_BWD_PROFILE={profile!r}")
    return selected


def case_tensors(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if case.dtype == "f16" else torch.float32
    gen_x = torch.Generator(device="cpu").manual_seed(8100 + sum(case.shape) * 17 + len(case.shape))
    gen_g = torch.Generator(device="cpu").manual_seed(9100 + sum(case.shape) * 19 + len(case.shape))
    x = (torch.randn(*case.shape, generator=gen_x, dtype=torch.float32) * 1.5).to(dtype).contiguous()
    g = (torch.randn(*case.shape, generator=gen_g, dtype=torch.float32) * 0.75).to(dtype).contiguous()
    return x, g


def dropout_mask(shape: tuple[int, ...], p: float, seed: int) -> torch.Tensor:
    n = int(np.prod(shape, dtype=np.int64))
    idx = np.arange(n, dtype=np.uint64)
    x = np.uint64(seed) + idx
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    mant = ((x >> np.uint64(40)) & np.uint64(0xFFFFFF)).astype(np.uint32)
    u = mant.astype(np.float32) * np.float32(1.0 / 16777216.0)
    return torch.from_numpy((u >= np.float32(p)).reshape(shape))


def ref_dropout_backward(g: torch.Tensor, case: Case) -> torch.Tensor:
    if case.p == 0.0:
        return g.clone()
    mask = dropout_mask(case.shape, case.p, case.seed).to(torch.bool)
    scale = 1.0 / (1.0 - case.p)
    if case.dtype == "f16":
        kept = (g.float() * scale).to(torch.float16)
    else:
        kept = g * scale
    return torch.where(mask, kept, torch.zeros_like(g))


def tensor_bytes(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous().numpy()
    if dtype == "f16":
        return arr.astype(np.float16, copy=False).view(np.uint16)
    return arr.astype(np.float32, copy=False)


def read_actual(path: Path, case: Case) -> torch.Tensor:
    if case.dtype == "f16":
        arr = np.fromfile(path, dtype=np.uint16).reshape(case.shape).view(np.float16)
        return torch.from_numpy(arr.copy()).to(torch.float16)
    arr = np.fromfile(path, dtype=np.float32).reshape(case.shape)
    return torch.from_numpy(arr.copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x, g = case_tensors(case)
    ref = ref_dropout_backward(g, case)
    x_path = tmp / f"{case.name}.x.bin"
    g_path = tmp / f"{case.name}.g.bin"
    dx_direct_path = tmp / f"{case.name}.dx_direct.bin"
    dx_auto_path = tmp / f"{case.name}.dx_auto.bin"
    tensor_bytes(x, case.dtype).tofile(x_path)
    tensor_bytes(g, case.dtype).tofile(g_path)
    env = gradients_env(root)
    cmd = [
        str(runner), str(x_path), str(g_path), str(dx_direct_path), str(dx_auto_path),
        case.dtype, str(case.p), str(case.seed), str(len(case.shape)), *(str(d) for d in case.shape),
    ]
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"dropout bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual_direct = read_actual(dx_direct_path, case)
    actual_auto = read_actual(dx_auto_path, case)
    tol = 1.0e-3 if case.dtype == "f16" else 1.0e-6
    direct_abs = float((actual_direct.float() - ref.float()).abs().max().item())
    auto_abs = float((actual_auto.float() - ref.float()).abs().max().item())
    ok = direct_abs <= tol and auto_abs <= tol
    print(
        f"dropout bwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} "
        f"p={case.p} direct_max_abs={direct_abs:.3e} auto_max_abs={auto_abs:.3e}"
    )
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-dropout-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_dropout_bwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"dropout bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
