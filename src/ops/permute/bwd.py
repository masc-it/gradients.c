# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c permute autograd correctness check against PyTorch."""

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
    axes: tuple[int, ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    size_t elem_size;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS] = {0};
    int64_t grad_shape[GD_MAX_DIMS] = {0};
    int32_t axes[GD_MAX_DIMS] = {0};
    uint32_t norm_axes[GD_MAX_DIMS] = {0};
    uint32_t seen = 0U;
    size_t count = 1U;
    size_t bytes;
    unsigned char *grad_host = NULL;
    unsigned char *dx_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, grad_out, y, dx;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 5 || parse_dtype_with_size(argv[2], &dtype, &elem_size) != 0 ||
        parse_u32(argv[3], &rank) != 0 || rank > GD_MAX_DIMS || argc != (int)(5U + rank * 2U)) {
        fprintf(stderr, "usage: %s DX.bin f16|f32 RANK X_DIMS... AXES... GRAD.bin\n", argv[0]);
        return 2;
    }
    arg = 4;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &shape[d]) != 0 || shape[d] <= 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)shape[d];
    }
    for (d = 0U; d < rank; ++d) {
        int32_t axis;
        uint32_t bit;
        if (parse_i32(argv[arg++], &axes[d]) != 0) { return 2; }
        axis = axes[d] < 0 ? axes[d] + (int32_t)rank : axes[d];
        if (axis < 0 || axis >= (int32_t)rank) { return 2; }
        bit = 1U << (uint32_t)axis;
        if ((seen & bit) != 0U) { return 2; }
        seen |= bit;
        norm_axes[d] = (uint32_t)axis;
        grad_shape[d] = shape[norm_axes[d]];
    }
    if (count > SIZE_MAX / elem_size) { return 2; }
    bytes = count * elem_size;
    grad_host = (unsigned char *)malloc(bytes);
    dx_host = (unsigned char *)malloc(bytes);
    if (grad_host == NULL || dx_host == NULL || read_file(argv[arg], grad_host, bytes) != 0) { goto fail; }

    oracle_memory_config_slots(&cfg,
                               align_up(bytes * 2U + 1024U * 1024U, 4096U),
                               align_up(bytes * 8U + 1024U * 1024U, 4096U),
                               4U,
                               2U);
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_zeros(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    x.requires_grad = true;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, grad_shape), 256U, &grad_out));
    CHECK(ctx, gd_tensor_write(ctx, &grad_out, grad_host, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_permute(ctx, &x, axes, rank, &y));
    CHECK(ctx, gd_backward(ctx, &y, &grad_out));
    CHECK(ctx, gd_tensor_grad(ctx, &x, &dx));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_host, bytes));
    if (write_file(argv[1], dx_host, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(grad_host);
    free(dx_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("matrix_f32", "f32", (2, 3), (1, 0)),
        Case("qkv_f16", "f16", (2, 4, 3, 8), (0, 2, 1, 3)),
        Case("mlp_f32", "f32", (4, 3, 5), (1, 0, 2)),
    ]
    profile = os.environ.get("GD_PERMUTE_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_PERMUTE_BWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if dtype == "f16" else torch.float32


def np_dtype(dtype: str) -> np.dtype:
    return np.float16 if dtype == "f16" else np.float32


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0xBADC0DE) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.zeros(case.shape, dtype=torch_dtype(case.dtype), requires_grad=True)
    y = x.permute(*case.axes)
    grad = (torch.randn(tuple(y.shape), generator=gen, dtype=torch.float32) * 0.2).to(torch_dtype(case.dtype))
    y.backward(grad)
    grad_path = tmp / f"{case.name}_grad.bin"
    dx_path = tmp / f"{case.name}_dx.bin"
    grad_path.write_bytes(grad.detach().contiguous().numpy().astype(np_dtype(case.dtype), copy=False).tobytes())
    cmd = [str(binary), str(dx_path), case.dtype, str(len(case.shape)),
           *map(str, case.shape), *map(str, case.axes), str(grad_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(dx_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(case.shape)
    want = x.grad.detach().numpy()
    atol = 1.0e-3 if case.dtype == "f16" else 1.0e-6
    np.testing.assert_allclose(got, want, rtol=0.0, atol=atol)
    print(f"[permute/bwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_permute_bwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
