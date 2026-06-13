# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c Huber backward correctness check against PyTorch."""

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
    grad_out: float


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
    float grad_value;
    unsigned char *x_data = NULL;
    unsigned char *y_data = NULL;
    unsigned char *dx_data = NULL;
    unsigned char *dy_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y, grad, dx, dy;
    gd_status st;
    uint32_t i;
    int rc = 1;

    if (argc < 9 || parse_dtype_with_size(argv[5], &dtype, &elem_size) != 0 || parse_f32(argv[6], &grad_value) != 0 ||
        parse_u32(argv[7], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS || argc != (int)(8U + rank)) {
        fprintf(stderr, "usage: %s X.bin Y.bin DX.bin DY.bin f16|f32 grad RANK DIM...\n", argv[0]);
        return 2;
    }
    for (i = 0U; i < rank; ++i) {
        if (parse_i64_dim(argv[8U + i], &shape[i]) != 0 ||
            (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / elems)) { return 2; }
        elems *= (size_t)shape[i];
    }
    if (elems == 0U || elems > SIZE_MAX / elem_size) { return 2; }
    bytes = elems * elem_size;
    x_data = (unsigned char *)malloc(bytes);
    y_data = (unsigned char *)malloc(bytes);
    dx_data = (unsigned char *)calloc(elems, elem_size);
    dy_data = (unsigned char *)calloc(elems, elem_size);
    if (x_data == NULL || y_data == NULL || dx_data == NULL || dy_data == NULL ||
        read_file(argv[1], x_data, bytes) != 0 || read_file(argv[2], y_data, bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(bytes * 2U + 1024U * 1024U, 4096U),
                         align_up(bytes * 4U + 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &y));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(0U, NULL), 256U, &grad));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &y, y_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &grad, &grad_value, sizeof(grad_value)));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_huber_backward(ctx, &x, &y, &grad, &dx, &dy));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, bytes));
    CHECK(ctx, gd_tensor_read(ctx, &dy, dy_data, bytes));
    if (write_file(argv[3], dx_data, bytes) != 0 || write_file(argv[4], dy_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(y_data); free(dx_data); free(dy_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    smoke = [
        Case("small_f16", "f16", (4, 7), 0.75),
        Case("tail_f16", "f16", (4099,), -1.25),
        Case("small_f32", "f32", (5, 13), 1.5),
    ]
    classic = [
        Case("activation_1024x1024_f16", "f16", (1024, 1024), 1.0),
        Case("segmentation_8x128x128_f16", "f16", (8, 128, 128), -0.5),
        Case("regression_64x1024_f32", "f32", (64, 1024), 2.0),
    ]
    profile = os.environ.get("GD_HUBER_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_HUBER_BWD_PROFILE={profile!r}")
    return selected


def case_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if case.dtype == "f16" else torch.float32
    gen = torch.Generator(device="cpu").manual_seed(8000 + len(case.shape) * 131 + sum(case.shape))
    x = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    y = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    return x, y


def tensor_bytes(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous().numpy()
    if dtype == "f16":
        return arr.astype(np.float16, copy=False).view(np.uint16)
    return arr.astype(np.float32, copy=False)


def read_tensor(path: Path, case: Case) -> torch.Tensor:
    if case.dtype == "f16":
        arr = np.fromfile(path, dtype=np.uint16).reshape(case.shape).view(np.float16)
        return torch.from_numpy(arr.copy()).to(torch.float16)
    arr = np.fromfile(path, dtype=np.float32).reshape(case.shape)
    return torch.from_numpy(arr.copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x, y = case_inputs(case)
    diff = x.float() - y.float()
    clipped = torch.clamp(diff, min=-1.0, max=1.0)
    ref_dx = clipped * (case.grad_out / x.numel())
    ref_dy = -ref_dx
    if case.dtype == "f16":
        ref_dx = ref_dx.to(torch.float16).float()
        ref_dy = ref_dy.to(torch.float16).float()
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    dx_path = tmp / f"{case.name}.dx.bin"
    dy_path = tmp / f"{case.name}.dy.bin"
    tensor_bytes(x, case.dtype).tofile(x_path)
    tensor_bytes(y, case.dtype).tofile(y_path)
    env = gradients_env(root)
    cmd = [str(runner), str(x_path), str(y_path), str(dx_path), str(dy_path),
           case.dtype, str(case.grad_out), str(len(case.shape)), *(str(d) for d in case.shape)]
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"huber bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    dx = read_tensor(dx_path, case).float()
    dy = read_tensor(dy_path, case).float()
    tol = 2.0e-3 if case.dtype == "f16" else 5.0e-6
    max_dx = float((dx - ref_dx).abs().max().item())
    max_dy = float((dy - ref_dy).abs().max().item())
    ok = max(max_dx, max_dy) <= tol
    print(f"huber bwd {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} dx={max_dx:.3e} dy={max_dy:.3e}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-huber-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_huber_bwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"huber bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
