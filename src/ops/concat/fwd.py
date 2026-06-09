# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c concat forward correctness check against PyTorch."""

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
    shapes: tuple[tuple[int, ...], ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#define MAX_INPUTS 8U

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    size_t elem_size;
    int32_t axis;
    uint32_t rank;
    uint32_t n_inputs;
    int64_t shapes[MAX_INPUTS][GD_MAX_DIMS];
    int64_t out_shape[GD_MAX_DIMS];
    size_t counts[MAX_INPUTS];
    size_t bytes[MAX_INPUTS];
    size_t out_count = 1U;
    size_t out_bytes;
    unsigned char *host[MAX_INPUTS] = {0};
    unsigned char *out_host = NULL;
    gd_tensor tensors[MAX_INPUTS];
    const gd_tensor *inputs[MAX_INPUTS];
    gd_tensor out;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    uint32_t i, d;
    int32_t norm_axis;
    int arg;
    int rc = 1;

    if (argc < 7 || parse_dtype_with_size(argv[2], &dtype, &elem_size) != 0 ||
        parse_i32(argv[3], &axis) != 0 || parse_u32(argv[4], &rank) != 0 ||
        parse_u32(argv[5], &n_inputs) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        n_inputs == 0U || n_inputs > MAX_INPUTS || argc != (int)(6U + n_inputs * rank + n_inputs)) {
        fprintf(stderr, "usage: %s OUT.bin f16|f32 AXIS RANK N_INPUTS DIMS... INPUTS...\n", argv[0]);
        return 2;
    }
    norm_axis = axis < 0 ? axis + (int32_t)rank : axis;
    if (norm_axis < 0 || norm_axis >= (int32_t)rank) { return 2; }
    arg = 6;
    for (i = 0U; i < n_inputs; ++i) {
        counts[i] = 1U;
        for (d = 0U; d < rank; ++d) {
            if (parse_i64_dim(argv[arg++], &shapes[i][d]) != 0 ||
                (uint64_t)shapes[i][d] > (uint64_t)(SIZE_MAX / counts[i])) { return 2; }
            counts[i] *= (size_t)shapes[i][d];
        }
        if (counts[i] > SIZE_MAX / elem_size) { return 2; }
        bytes[i] = counts[i] * elem_size;
    }
    for (d = 0U; d < rank; ++d) { out_shape[d] = shapes[0][d]; }
    out_shape[norm_axis] = 0;
    for (i = 0U; i < n_inputs; ++i) {
        for (d = 0U; d < rank; ++d) {
            if (d != (uint32_t)norm_axis && shapes[i][d] != shapes[0][d]) { return 2; }
        }
        out_shape[norm_axis] += shapes[i][norm_axis];
    }
    for (d = 0U; d < rank; ++d) {
        if ((uint64_t)out_shape[d] > (uint64_t)(SIZE_MAX / out_count)) { return 2; }
        out_count *= (size_t)out_shape[d];
    }
    if (out_count > SIZE_MAX / elem_size) { return 2; }
    out_bytes = out_count * elem_size;
    for (i = 0U; i < n_inputs; ++i) {
        host[i] = (unsigned char *)malloc(bytes[i]);
        if (host[i] == NULL || read_file(argv[arg++], host[i], bytes[i]) != 0) { goto fail; }
    }
    out_host = (unsigned char *)malloc(out_bytes);
    if (out_host == NULL) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(out_bytes + 1024U * 1024U, 4096U),
                         align_up(out_bytes * 2U + 1024U * 1024U, 4096U));
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    for (i = 0U; i < n_inputs; ++i) {
        CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shapes[i]), 256U, &tensors[i]));
        CHECK(ctx, gd_tensor_write(ctx, &tensors[i], host[i], bytes[i]));
        inputs[i] = &tensors[i];
    }
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK(ctx, gd_concat(ctx, inputs, n_inputs, axis, &out));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &out, out_host, out_bytes));
    if (write_file(argv[1], out_host, out_bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    for (i = 0U; i < MAX_INPUTS; ++i) { free(host[i]); }
    free(out_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("axis0_f16", "f16", 0, ((3, 5), (2, 5))),
        Case("axis1_f16", "f16", 1, ((4, 7), (4, 3), (4, 5))),
        Case("rank3_neg_axis_f32", "f32", -1, ((2, 3, 4), (2, 3, 5))),
        Case("features_f32", "f32", 1, ((128, 256), (128, 512))),
    ]
    profile = os.environ.get("GD_CONCAT_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:3]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_CONCAT_FWD_PROFILE={profile!r}")
    return selected


def np_dtype(dtype: str) -> np.dtype:
    return np.float16 if dtype == "f16" else np.float32


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(0.0, 0.2, shape).astype(np_dtype(case.dtype)) for shape in case.shapes]
    paths = []
    for i, arr in enumerate(arrays):
        path = tmp / f"{case.name}_{i}.bin"
        path.write_bytes(arr.tobytes())
        paths.append(path)
    out_path = tmp / f"{case.name}_out.bin"
    dims = [str(dim) for shape in case.shapes for dim in shape]
    cmd = [str(binary), str(out_path), case.dtype, str(case.axis), str(len(case.shapes[0])),
           str(len(case.shapes)), *dims, *map(str, paths)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(out_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(
        torch.cat([torch.from_numpy(a) for a in arrays], dim=case.axis).shape
    )
    want = torch.cat([torch.from_numpy(a) for a in arrays], dim=case.axis).numpy()
    np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=0, atol=0)
    print(f"[concat/fwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_concat_fwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
