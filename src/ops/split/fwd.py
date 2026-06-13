# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c split forward correctness check against PyTorch."""

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
#define PATH_MAX_LOCAL 4096U

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
    unsigned char *host = NULL;
    unsigned char *tmp_host = NULL;
    gd_tensor x;
    gd_tensor outputs[MAX_OUTPUTS];
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    uint32_t i, d;
    int32_t norm_axis;
    int arg;
    int rc = 1;

    if (argc < 8 || parse_dtype_with_size(argv[2], &dtype, &elem_size) != 0 ||
        parse_i32(argv[3], &axis) != 0 || parse_u32(argv[4], &rank) != 0 ||
        parse_u32(argv[5], &n_outputs) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        n_outputs == 0U || n_outputs > MAX_OUTPUTS || argc != (int)(7U + rank + n_outputs)) {
        fprintf(stderr, "usage: %s OUT_PREFIX u8|f16|f32 AXIS RANK N_OUTPUTS DIMS... SIZES... INPUT.bin\n", argv[0]);
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
    host = (unsigned char *)malloc(input_bytes);
    tmp_host = (unsigned char *)malloc(input_bytes);
    if (host == NULL || tmp_host == NULL || read_file(argv[arg], host, input_bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(input_bytes + 1024U * 1024U, 4096U),
                         align_up(input_bytes * 3U + 1024U * 1024U, 4096U));
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, host, input_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK(ctx, gd_split(ctx, &x, sizes, n_outputs, axis, outputs));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    for (i = 0U; i < n_outputs; ++i) {
        char path[PATH_MAX_LOCAL];
        size_t count = input_count / (size_t)shape[norm_axis] * (size_t)sizes[i];
        size_t bytes = count * elem_size;
        int n = snprintf(path, sizeof(path), "%s_%u.bin", argv[1], i);
        if (n < 0 || (size_t)n >= sizeof(path)) { goto fail; }
        CHECK(ctx, gd_tensor_read(ctx, &outputs[i], tmp_host, bytes));
        if (write_file(path, tmp_host, bytes) != 0) { goto fail; }
    }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(host);
    free(tmp_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("qkv_axis2_f16", "f16", 2, (2, 8, 3, 4, 16), (1, 1, 1)),
        Case("hidden_last_f32", "f32", -1, (4, 17), (5, 7, 5)),
        Case("image_channels_u8", "u8", -1, (6, 5, 3), (1, 2)),
        Case("qkv_flat_f16", "f16", -1, (2, 64, 768), (256, 256, 256)),
    ]
    profile = os.environ.get("GD_SPLIT_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:3]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_SPLIT_FWD_PROFILE={profile!r}")
    return selected


def np_dtype(dtype: str) -> np.dtype:
    if dtype == "u8":
        return np.uint8
    return np.float16 if dtype == "f16" else np.float32


def torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "u8":
        return torch.uint8
    return torch.float16 if dtype == "f16" else torch.float32


def make_input(case: Case) -> np.ndarray:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    if case.dtype == "u8":
        return rng.integers(0, 251, size=case.shape, dtype=np.uint8)
    return rng.normal(0.0, 0.2, case.shape).astype(np_dtype(case.dtype))


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x_np = make_input(case)
    x_path = tmp / f"{case.name}_x.bin"
    x_path.write_bytes(x_np.tobytes())
    prefix = tmp / f"{case.name}_out"
    cmd = [str(binary), str(prefix), case.dtype, str(case.axis), str(len(case.shape)), str(len(case.sizes)),
           *[str(dim) for dim in case.shape], *[str(size) for size in case.sizes], str(x_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    want = torch.split(torch.from_numpy(x_np).to(torch_dtype(case.dtype)), list(case.sizes), dim=case.axis)
    for i, expected in enumerate(want):
        got = np.frombuffer((tmp / f"{case.name}_out_{i}.bin").read_bytes(), dtype=np_dtype(case.dtype)).reshape(tuple(expected.shape))
        if case.dtype == "u8":
            np.testing.assert_array_equal(got, expected.numpy())
        else:
            np.testing.assert_allclose(got.astype(np.float32), expected.numpy().astype(np.float32), rtol=0, atol=0)
    print(f"[split/fwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_split_fwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
