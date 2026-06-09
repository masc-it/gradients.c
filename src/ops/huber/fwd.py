# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c Huber forward correctness check against PyTorch."""

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
    unsigned char *y_data = NULL;
    float loss_value = 0.0f;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y, loss;
    gd_status st;
    uint32_t i;
    int rc = 1;

    if (argc < 7 || parse_dtype_with_size(argv[4], &dtype, &elem_size) != 0 || parse_u32(argv[5], &rank) != 0 ||
        rank == 0U || rank > GD_MAX_DIMS || argc != (int)(6U + rank)) {
        fprintf(stderr, "usage: %s X.bin Y.bin OUT.bin f16|f32 RANK DIM...\n", argv[0]);
        return 2;
    }
    for (i = 0U; i < rank; ++i) {
        if (parse_i64_dim(argv[6U + i], &shape[i]) != 0 ||
            (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / elems)) { return 2; }
        elems *= (size_t)shape[i];
    }
    if (elems == 0U || elems > SIZE_MAX / elem_size) { return 2; }
    bytes = elems * elem_size;
    x_data = (unsigned char *)malloc(bytes);
    y_data = (unsigned char *)malloc(bytes);
    if (x_data == NULL || y_data == NULL || read_file(argv[1], x_data, bytes) != 0 ||
        read_file(argv[2], y_data, bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(bytes * 2U + 1024U * 1024U, 4096U),
                         align_up(bytes * 2U + 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &y));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &y, y_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_huber(ctx, &x, &y, &loss));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &loss, &loss_value, sizeof(loss_value)));
    if (write_file(argv[3], &loss_value, sizeof(loss_value)) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(y_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    smoke = [
        Case("small_f16", "f16", (4, 7)),
        Case("tail_partial_f16", "f16", (3, 1371)),
        Case("small_f32", "f32", (5, 13)),
    ]
    classic = [
        Case("activation_1024x1024_f16", "f16", (1024, 1024)),
        Case("segmentation_8x256x256_f16", "f16", (8, 256, 256)),
        Case("regression_64x1024_f32", "f32", (64, 1024)),
    ]
    profile = os.environ.get("GD_HUBER_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_HUBER_FWD_PROFILE={profile!r}")
    return selected


def case_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if case.dtype == "f16" else torch.float32
    gen = torch.Generator(device="cpu").manual_seed(7000 + len(case.shape) * 97 + sum(case.shape))
    x = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    y = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    return x, y


def tensor_bytes(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous().numpy()
    if dtype == "f16":
        return arr.astype(np.float16, copy=False).view(np.uint16)
    return arr.astype(np.float32, copy=False)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x, y = case_inputs(case)
    ref = torch.nn.functional.huber_loss(x.float(), y.float(), reduction="mean", delta=1.0)
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    out_path = tmp / f"{case.name}.out.bin"
    tensor_bytes(x, case.dtype).tofile(x_path)
    tensor_bytes(y, case.dtype).tofile(y_path)
    env = gradients_env(root)
    cmd = [str(runner), str(x_path), str(y_path), str(out_path), case.dtype,
           str(len(case.shape)), *(str(d) for d in case.shape)]
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"huber fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual = float(np.fromfile(out_path, dtype=np.float32)[0])
    tol = 4.0e-4 if case.dtype == "f16" else 4.0e-6
    max_abs = abs(actual - float(ref.item()))
    ok = max_abs <= tol
    print(f"huber fwd {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-huber-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_huber_fwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"huber fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
