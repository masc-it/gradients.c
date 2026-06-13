# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c FP16 ReLU forward correctness check against PyTorch."""

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
    rows: int
    cols: int
    edge: bool = False


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    uint32_t rows, cols;
    size_t elems, bytes;
    uint16_t *x_data = NULL;
    uint16_t *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y;
    int64_t shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 5 || parse_u32(argv[3], &rows) != 0 || parse_u32(argv[4], &cols) != 0) {
        fprintf(stderr, "usage: %s X.bin Y.bin ROWS COLS\n", argv[0]);
        return 2;
    }
    elems = (size_t)rows * (size_t)cols;
    bytes = elems * sizeof(uint16_t);
    x_data = (uint16_t *)malloc(bytes);
    y_data = (uint16_t *)calloc(elems, sizeof(uint16_t));
    if (x_data == NULL || y_data == NULL || read_file(argv[1], x_data, bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(bytes + 1024U * 1024U, 4096U),
                         align_up(bytes + 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    shape[0] = (int64_t)rows; shape[1] = (int64_t)cols;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_relu(ctx, &x, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &y, y_data, bytes));
    if (write_file(argv[2], y_data, bytes) != 0) { goto fail; }
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
    smoke = [Case("edge_f16", 1, 8, True), Case("small_tail_f16", 4, 17)]
    classic = [Case("large_square_f16", 1024, 1024), Case("large_tail_f16", 2048, 513)]
    profile = os.environ.get("GD_RELU_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RELU_FWD_PROFILE={profile!r}")
    return selected


def edge_input(rows: int, cols: int) -> torch.Tensor:
    bits = np.array([0x7E00, 0x8000, 0x0000, 0xBC00, 0x3C00, 0x7C00, 0xFC00, 0x0001], dtype=np.uint16)
    if rows * cols != bits.size:
        raise ValueError("edge case shape must match edge vector")
    return torch.from_numpy(bits.view(np.float16).reshape(rows, cols).copy())


def case_input(case: Case) -> torch.Tensor:
    if case.edge:
        return edge_input(case.rows, case.cols)
    gen = torch.Generator(device="cpu").manual_seed(1000 + case.rows * 3 + case.cols)
    return (torch.randn(case.rows, case.cols, generator=gen, dtype=torch.float32) * 2.0 - 0.2).to(torch.float16)


def f16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy().astype(np.float16, copy=False).view(np.uint16)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x = case_input(case)
    ref = torch.relu(x)
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    f16_bits(x).tofile(x_path)
    env = gradients_env(root)
    proc = subprocess.run([str(runner), str(x_path), str(y_path), str(case.rows), str(case.cols)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"relu fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual_bits = np.fromfile(y_path, dtype=np.uint16).reshape(case.rows, case.cols)
    ref_bits = f16_bits(ref).reshape(case.rows, case.cols)
    ok = bool(np.array_equal(actual_bits, ref_bits))
    mismatches = int(np.count_nonzero(actual_bits != ref_bits))
    print(f"relu fwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype=f16 mismatches={mismatches}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-relu-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_relu_fwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"relu fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
