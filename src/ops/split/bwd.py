# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c split backward correctness check against PyTorch autograd."""

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
    axis: int
    shape: tuple[int, ...]
    sizes: tuple[int, ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#define MAX_OUTPUTS 8U

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    size_t elem_size;
    int32_t axis;
    uint32_t rank;
    uint32_t n_outputs;
    int64_t shape[GD_MAX_DIMS];
    int64_t sizes[MAX_OUTPUTS];
    size_t input_count = 1U;
    size_t input_bytes;
    unsigned char *tmp_host = NULL;
    gd_tensor x;
    gd_tensor grad_outputs_storage[MAX_OUTPUTS];
    const gd_tensor *grad_outputs[MAX_OUTPUTS];
    gd_tensor dx;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    uint32_t i, d;
    int32_t norm_axis;
    int arg;
    int rc = 1;

    if (argc < 8 || parse_dtype_with_size(argv[2], &dtype, &elem_size) != 0 ||
        parse_i32(argv[3], &axis) != 0 || parse_u32(argv[4], &rank) != 0 ||
        parse_u32(argv[5], &n_outputs) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        n_outputs == 0U || n_outputs > MAX_OUTPUTS || argc != (int)(6U + rank + n_outputs * 2U)) {
        fprintf(stderr, "usage: %s DX.bin f16|f32 AXIS RANK N_OUTPUTS DIMS... SIZES... GRAD_BINS...\n", argv[0]);
        return 2;
    }
    norm_axis = axis < 0 ? axis + (int32_t)rank : axis;
    if (norm_axis < 0 || norm_axis >= (int32_t)rank) { return 2; }
    arg = 6;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64_dim(argv[arg++], &shape[d]) != 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / input_count)) { return 2; }
        input_count *= (size_t)shape[d];
    }
    for (i = 0U; i < n_outputs; ++i) {
        if (parse_i64_dim(argv[arg++], &sizes[i]) != 0) { return 2; }
    }
    if (input_count > SIZE_MAX / elem_size) { return 2; }
    input_bytes = input_count * elem_size;
    tmp_host = (unsigned char *)malloc(input_bytes);
    if (tmp_host == NULL) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(input_bytes * 2U + 1024U * 1024U, 4096U),
                         align_up(input_bytes * 4U + 1024U * 1024U, 4096U));
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_zeros(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    for (i = 0U; i < n_outputs; ++i) {
        int64_t gshape[GD_MAX_DIMS];
        size_t count;
        size_t bytes;
        for (d = 0U; d < rank; ++d) { gshape[d] = shape[d]; }
        gshape[norm_axis] = sizes[i];
        count = input_count / (size_t)shape[norm_axis] * (size_t)sizes[i];
        bytes = count * elem_size;
        if (read_file(argv[arg++], tmp_host, bytes) != 0) { goto fail; }
        CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, gshape), 256U, &grad_outputs_storage[i]));
        CHECK(ctx, gd_tensor_write(ctx, &grad_outputs_storage[i], tmp_host, bytes));
        grad_outputs[i] = &grad_outputs_storage[i];
    }
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_split_backward(ctx, &x, grad_outputs, sizes, n_outputs, axis, &dx));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, tmp_host, input_bytes));
    if (write_file(argv[1], tmp_host, input_bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(tmp_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("qkv_axis2_f16", "f16", 2, (2, 8, 3, 4, 16), (1, 1, 1)),
        Case("hidden_last_f32", "f32", -1, (4, 17), (5, 7, 5)),
        Case("qkv_flat_f16", "f16", -1, (2, 64, 768), (256, 256, 256)),
    ]
    profile = os.environ.get("GD_SPLIT_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_SPLIT_BWD_PROFILE={profile!r}")
    return selected


def np_dtype(dtype: str) -> np.dtype:
    return np.float16 if dtype == "f16" else np.float32


def torch_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if dtype == "f16" else torch.float32


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x51A17) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    x = torch.zeros(case.shape, dtype=torch_dtype(case.dtype), requires_grad=True)
    outs = torch.split(x, list(case.sizes), dim=case.axis)
    grad_arrays = [rng.normal(0.0, 0.2, tuple(out.shape)).astype(np_dtype(case.dtype)) for out in outs]
    torch.autograd.backward(outs, [torch.from_numpy(arr) for arr in grad_arrays])
    grad_paths = []
    for i, arr in enumerate(grad_arrays):
        path = tmp / f"{case.name}_g{i}.bin"
        path.write_bytes(arr.tobytes())
        grad_paths.append(path)
    dx_path = tmp / f"{case.name}_dx.bin"
    cmd = [str(binary), str(dx_path), case.dtype, str(case.axis), str(len(case.shape)), str(len(case.sizes)),
           *[str(dim) for dim in case.shape], *[str(size) for size in case.sizes], *map(str, grad_paths)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(dx_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(case.shape)
    want = x.grad.detach().numpy()
    np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=0, atol=0)
    print(f"[split/bwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_split_bwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
