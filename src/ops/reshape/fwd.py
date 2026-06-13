# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c reshape forward correctness check against PyTorch."""

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
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    size_t elem_size;
    uint32_t input_rank;
    uint32_t target_rank;
    int64_t input_shape[GD_MAX_DIMS] = {0};
    int64_t target_shape[GD_MAX_DIMS] = {0};
    size_t count = 1U;
    size_t bytes;
    unsigned char *host = NULL;
    unsigned char *out_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 7 || parse_dtype_with_size(argv[2], &dtype, &elem_size) != 0 ||
        parse_u32(argv[3], &input_rank) != 0 || parse_u32(argv[4], &target_rank) != 0 ||
        input_rank > GD_MAX_DIMS || target_rank > GD_MAX_DIMS ||
        argc != (int)(6U + input_rank + target_rank)) {
        fprintf(stderr, "usage: %s OUT.bin f16|f32|u8 INPUT_RANK TARGET_RANK INPUT_DIMS... TARGET_DIMS... X.bin\n", argv[0]);
        return 2;
    }
    arg = 5;
    for (d = 0U; d < input_rank; ++d) {
        if (parse_i64(argv[arg++], &input_shape[d]) != 0 || input_shape[d] <= 0 ||
            (uint64_t)input_shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)input_shape[d];
    }
    for (d = 0U; d < target_rank; ++d) {
        if (parse_i64(argv[arg++], &target_shape[d]) != 0) { return 2; }
    }
    if (count > SIZE_MAX / elem_size) { return 2; }
    bytes = count * elem_size;
    host = (unsigned char *)malloc(bytes);
    out_host = (unsigned char *)malloc(bytes);
    if (host == NULL || out_host == NULL || read_file(argv[arg], host, bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(bytes + 1024U * 1024U, 4096U),
                         align_up(bytes * 2U + 1024U * 1024U, 4096U));
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(input_rank, input_shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, host, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK(ctx, gd_reshape(ctx, &x, gd_shape_make(target_rank, target_shape), &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &y, out_host, bytes));
    if (write_file(argv[1], out_host, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(host);
    free(out_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("matrix_f16", "f16", (2, 3), (3, 2)),
        Case("infer_rank3_f32", "f32", (2, 3, 4), (2, -1, 3)),
        Case("ids_u8", "u8", (4, 5), (-1,)),
        Case("heads_f16", "f16", (2, 4, 8, 16), (2, 8, -1)),
    ]
    profile = os.environ.get("GD_RESHAPE_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:3]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RESHAPE_FWD_PROFILE={profile!r}")
    return selected


def np_dtype(dtype: str) -> np.dtype:
    if dtype == "f16":
        return np.float16
    if dtype == "f32":
        return np.float32
    return np.uint8


def make_input(case: Case) -> np.ndarray:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    if case.dtype == "u8":
        return rng.integers(0, 255, size=case.input_shape, dtype=np.uint8)
    return rng.normal(0.0, 0.2, case.input_shape).astype(np_dtype(case.dtype))


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x = make_input(case)
    ref = torch.from_numpy(x).reshape(*case.target_shape).contiguous().numpy()
    x_path = tmp / f"{case.name}_x.bin"
    out_path = tmp / f"{case.name}_out.bin"
    x_path.write_bytes(np.ascontiguousarray(x).tobytes())
    cmd = [str(binary), str(out_path), case.dtype, str(len(case.input_shape)), str(len(case.target_shape)),
           *map(str, case.input_shape), *map(str, case.target_shape), str(x_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(out_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(tuple(ref.shape))
    np.testing.assert_array_equal(got, ref)
    print(f"[reshape/fwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_reshape_fwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
