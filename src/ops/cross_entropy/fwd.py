# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c cross_entropy forward correctness check against PyTorch."""

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
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    classes: int
    dtype: str


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    uint32_t rows, classes, dtype;
    size_t elems, elem_size, logits_bytes, label_bytes;
    void *logits_data = NULL;
    int32_t *target_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor logits, targets, loss;
    int64_t logits_shape[2];
    int64_t target_shape[1];
    float loss_value = 0.0f;
    gd_status st;
    int rc = 1;

    if (argc != 7 || parse_u32(argv[4], &rows) != 0 || parse_u32(argv[5], &classes) != 0 ||
        parse_u32(argv[6], &dtype) != 0) {
        fprintf(stderr, "usage: %s LOGITS.bin TARGETS.bin LOSS.bin ROWS CLASSES DTYPE\n", argv[0]);
        return 2;
    }
    elem_size = dtype == (uint32_t)GD_DTYPE_F32 ? 4U : 2U;
    elems = (size_t)rows * (size_t)classes;
    logits_bytes = elems * elem_size;
    label_bytes = (size_t)rows * sizeof(*target_data);
    logits_data = malloc(logits_bytes);
    target_data = (int32_t *)malloc(label_bytes);
    if (logits_data == NULL || target_data == NULL ||
        read_file(argv[1], logits_data, logits_bytes) != 0 ||
        read_file(argv[2], target_data, label_bytes) != 0) { goto fail; }

    oracle_memory_config(&cfg,
                         align_up(logits_bytes + label_bytes + 1024U * 1024U, 4096U),
                         align_up(logits_bytes * 2U + label_bytes + 16U * 1024U * 1024U, 4096U));
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }

    logits_shape[0] = (int64_t)rows; logits_shape[1] = (int64_t)classes;
    target_shape[0] = (int64_t)rows;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, (gd_dtype)dtype, gd_shape_make(2U, logits_shape), 256U, &logits));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, target_shape), 256U, &targets));
    CHECK(ctx, gd_tensor_write(ctx, &logits, logits_data, logits_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &targets, target_data, label_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK(ctx, gd_cross_entropy(ctx, &logits, &targets, &loss));
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
    free(logits_data); free(target_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    smoke = [Case("small_f16", 3, 11, "f16")]
    classic = [Case("classes2048_f16", 64, 2048, "f16")]
    profile = os.environ.get("GD_CROSS_ENTROPY_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_CROSS_ENTROPY_FWD_PROFILE={profile!r}")
    return selected


def dtype_info(name: str) -> tuple[torch.dtype, np.dtype, int]:
    if name == "f16":
        return torch.float16, np.float16, 1
    if name == "f32":
        return torch.float32, np.float32, 3
    raise ValueError(name)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    torch_dtype, np_dtype, gd_dtype = dtype_info(case.dtype)
    gen = torch.Generator(device="cpu").manual_seed(3100 + case.rows * 17 + case.classes)
    logits = (torch.randn(case.rows, case.classes, generator=gen, dtype=torch.float32) * 1.7).to(torch_dtype)
    targets = torch.randint(0, case.classes, (case.rows,), generator=gen, dtype=torch.long)
    ref_logits = logits.to(torch.float32)
    ref = F.cross_entropy(ref_logits, targets, reduction="mean")

    logits_path = tmp / f"{case.name}.logits.bin"
    targets_path = tmp / f"{case.name}.targets.bin"
    loss_path = tmp / f"{case.name}.loss.bin"
    logits.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).tofile(logits_path)
    targets.cpu().numpy().astype(np.int32, copy=False).tofile(targets_path)

    env = gradients_env(root)
    proc = subprocess.run([str(runner), str(logits_path), str(targets_path), str(loss_path),
                           str(case.rows), str(case.classes), str(gd_dtype)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"cross_entropy fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual = float(np.fromfile(loss_path, dtype=np.float32)[0])
    want = float(ref.item())
    max_abs = abs(actual - want)
    tol = 2.0e-3 if case.dtype == "f16" else 2.0e-4
    ok = max_abs <= tol
    print(f"cross_entropy fwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} "
          f"loss={actual:.6f} ref={want:.6f} max_abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-cross-entropy-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_cross_entropy_fwd_runner", RUNNER_SOURCE)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"cross_entropy fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
