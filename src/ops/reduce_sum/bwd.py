# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c reduce_sum backward/autograd correctness check against PyTorch."""

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

AXIS_ALL = -999


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    dtype: str
    axis: int = AXIS_ALL


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#define AXIS_ALL (-999)

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    int32_t dtype_i, rank_i, axis;
    gd_dtype dtype;
    uint32_t rank;
    uint32_t grad_rank;
    int64_t shape[GD_MAX_DIMS];
    int64_t grad_shape[GD_MAX_DIMS];
    size_t x_count, grad_count, elem_size, x_bytes, grad_bytes;
    void *x_data = NULL;
    void *grad_data = NULL;
    void *dx_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, grad_out, out, dx;
    gd_status st;
    int rc = 1;
    uint32_t i;

    if (argc < 8 || parse_i32(argv[4], &dtype_i) != 0 || parse_i32(argv[5], &rank_i) != 0 ||
        parse_i32(argv[6], &axis) != 0 || rank_i <= 0 || rank_i > (int32_t)GD_MAX_DIMS || argc != 7 + rank_i) {
        fprintf(stderr, "usage: %s X.bin GRAD.bin DX.bin DTYPE RANK AXIS DIMS...\n", argv[0]);
        return 2;
    }
    rank = (uint32_t)rank_i;
    for (i = 0U; i < rank; ++i) {
        if (parse_i64(argv[7 + i], &shape[i]) != 0) { return 2; }
    }
    dtype = (gd_dtype)dtype_i;
    elem_size = dtype == GD_DTYPE_F32 ? 4U : 2U;
    if (tensor_count(rank, shape, &x_count) != 0 ||
        axis_shape(rank, shape, axis, &grad_rank, grad_shape) != 0 ||
        tensor_count(grad_rank, grad_shape, &grad_count) != 0) { return 2; }
    x_bytes = x_count * elem_size;
    grad_bytes = grad_count * elem_size;
    x_data = malloc(x_bytes);
    grad_data = malloc(grad_bytes);
    dx_data = calloc(x_count, elem_size);
    if (x_data == NULL || grad_data == NULL || dx_data == NULL ||
        read_file(argv[1], x_data, x_bytes) != 0 || read_file(argv[2], grad_data, grad_bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(x_bytes + grad_bytes + 1024U * 1024U, 4096U),
                         align_up(x_bytes * 6U + grad_bytes * 2U + 32U * 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(grad_rank, grad_rank == 0U ? NULL : grad_shape), 256U, &grad_out));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &grad_out, grad_data, grad_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    x.requires_grad = true;
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    if (axis == AXIS_ALL) {
        CHECK(ctx, gd_reduce_sum(ctx, &x, &out));
    } else {
        CHECK(ctx, gd_reduce_sum_axis(ctx, &x, axis, false, &out));
    }
    CHECK(ctx, gd_backward(ctx, &out, &grad_out));
    CHECK(ctx, gd_tensor_grad(ctx, &x, &dx));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, x_bytes));
    if (write_file(argv[3], dx_data, x_bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(grad_data); free(dx_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    smoke = [
        Case("all_tail_f16", (513,), "f16"),
        Case("last_axis_f16", (4, 257), "f16", -1),
        Case("axis0_f32", (7, 19), "f32", 0),
    ]
    classic = [
        Case("all_1024x1024_f16", (1024, 1024), "f16"),
        Case("last_256x513_f16", (256, 513), "f16", -1),
        Case("middle_8x33x65_f16", (8, 33, 65), "f16", 1),
        Case("all_1024x1024_f32", (1024, 1024), "f32"),
    ]
    profile = os.environ.get("GD_REDUCE_SUM_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_REDUCE_SUM_BWD_PROFILE={profile!r}")
    return selected


def dtype_info(name: str) -> tuple[torch.dtype, np.dtype, int]:
    if name == "f16":
        return torch.float16, np.float16, 1
    if name == "f32":
        return torch.float32, np.float32, 3
    raise ValueError(name)


def output_shape(case: Case) -> tuple[int, ...]:
    if case.axis == AXIS_ALL:
        return ()
    axis = case.axis if case.axis >= 0 else case.axis + len(case.shape)
    return tuple(dim for i, dim in enumerate(case.shape) if i != axis)


def broadcast_grad(grad: torch.Tensor, case: Case) -> torch.Tensor:
    if case.axis == AXIS_ALL:
        return grad.reshape(()).expand(case.shape)
    axis = case.axis if case.axis >= 0 else case.axis + len(case.shape)
    return grad.unsqueeze(axis).expand(case.shape)


def tensor_to_file(t: torch.Tensor, path: Path, np_dtype: np.dtype) -> None:
    t.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).tofile(path)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    torch_dtype, np_dtype, gd_dtype = dtype_info(case.dtype)
    gen = torch.Generator(device="cpu").manual_seed(6100 + sum(case.shape) * 11 + len(case.shape))
    x = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 0.75).to(torch_dtype)
    grad_shape = output_shape(case)
    if grad_shape == ():
        grad = (torch.randn((), generator=gen, dtype=torch.float32) * 0.5).to(torch_dtype)
    else:
        grad = (torch.randn(*grad_shape, generator=gen, dtype=torch.float32) * 0.5).to(torch_dtype)
    ref = broadcast_grad(grad, case).to(torch_dtype if case.dtype == "f16" else torch.float32)
    x_path = tmp / f"{case.name}.x.bin"
    grad_path = tmp / f"{case.name}.grad.bin"
    dx_path = tmp / f"{case.name}.dx.bin"
    tensor_to_file(x, x_path, np_dtype)
    tensor_to_file(grad, grad_path, np_dtype)
    env = gradients_env(root)
    proc = subprocess.run([str(runner), str(x_path), str(grad_path), str(dx_path), str(gd_dtype),
                           str(len(case.shape)), str(case.axis), *[str(dim) for dim in case.shape]],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"reduce_sum bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual = np.fromfile(dx_path, dtype=np_dtype).reshape(case.shape)
    want = ref.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).reshape(case.shape)
    actual32 = actual.astype(np.float32, copy=False)
    want32 = want.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(actual32 - want32)))
    tol = 0.0 if case.dtype == "f16" else 1.0e-7
    ok = max_abs <= tol
    print(f"reduce_sum bwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} "
          f"shape={case.shape} axis={case.axis} max_abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-reduce-sum-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_reduce_sum_bwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"reduce_sum bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
