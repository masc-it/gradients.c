# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Linear forward reference + live gradients.c correctness check."""

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
    m: int
    k: int
    n: int


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    uint32_t m, k, n, has_bias;
    size_t x_elems, w_elems, y_elems, x_bytes, w_bytes, b_bytes, y_bytes;
    uint16_t *x_data = NULL, *w_data = NULL, *b_data = NULL, *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, b, y;
    int64_t x_shape[2], w_shape[2], b_shape[1];
    gd_status st;
    int rc = 1;

    if (argc != 8) {
        fprintf(stderr, "usage: %s X.bin W.bin B.bin|- Y.bin M K N\n", argv[0]);
        return 2;
    }
    if (parse_u32(argv[5], &m) != 0 || parse_u32(argv[6], &k) != 0 ||
        parse_u32(argv[7], &n) != 0) { return 2; }
    has_bias = strcmp(argv[3], "-") != 0 ? 1U : 0U;
    if (mul_size((size_t)m, (size_t)k, &x_elems) != 0 ||
        mul_size((size_t)k, (size_t)n, &w_elems) != 0 ||
        mul_size((size_t)m, (size_t)n, &y_elems) != 0 ||
        mul_size(x_elems, 2U, &x_bytes) != 0 || mul_size(w_elems, 2U, &w_bytes) != 0 ||
        mul_size(y_elems, 2U, &y_bytes) != 0 || mul_size((size_t)n, 2U, &b_bytes) != 0) { return 2; }

    x_data = (uint16_t *)malloc(x_bytes);
    w_data = (uint16_t *)malloc(w_bytes);
    b_data = (uint16_t *)malloc(b_bytes);
    y_data = (uint16_t *)calloc(y_elems, sizeof(*y_data));
    if (x_data == NULL || w_data == NULL || b_data == NULL || y_data == NULL) { goto fail; }
    if (read_file(argv[1], x_data, x_bytes) != 0 || read_file(argv[2], w_data, w_bytes) != 0 ||
        (has_bias != 0U && read_file(argv[3], b_data, b_bytes) != 0)) { goto fail; }

    oracle_memory_config_slots(&cfg,
                               align_up(x_bytes + w_bytes + b_bytes + 16U * 1024U * 1024U, 4096U),
                               align_up(y_bytes + 16U * 1024U * 1024U, 4096U),
                               2U,
                               2U);

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }

    x_shape[0] = (int64_t)m; x_shape[1] = (int64_t)k;
    w_shape[0] = (int64_t)k; w_shape[1] = (int64_t)n;
    b_shape[0] = (int64_t)n;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &w, w_data, w_bytes));
    if (has_bias != 0U) {
        CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, b_shape), 256U, &b));
        CHECK(ctx, gd_tensor_write(ctx, &b, b_data, b_bytes));
    }
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_linear(ctx, &x, &w, has_bias != 0U ? &b : NULL, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &y, y_data, y_bytes));
    if (write_file(argv[4], y_data, y_bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(w_data); free(b_data); free(y_data);
    return rc;
}
'''


def make_cases() -> list[Case]:
    tokens = int(os.environ.get("GD_LINEAR_FWD_TOKENS", "512"))
    smoke = [Case("fallback_small", 4, 7, 6)]
    classic: list[Case] = []
    for hidden in (128, 256, 512):
        intermediate = hidden * 4
        classic.extend([
            Case(f"h{hidden}_qkv", tokens, hidden, 3 * hidden),
            Case(f"h{hidden}_proj", tokens, hidden, hidden),
            Case(f"h{hidden}_mlp_up", tokens, hidden, intermediate),
            Case(f"h{hidden}_mlp_down", tokens, intermediate, hidden),
        ])
    profile = os.environ.get("GD_LINEAR_FWD_PROFILE", "classic")
    if profile == "smoke":
        return smoke
    if profile in {"classic", "all"}:
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_LINEAR_FWD_PROFILE={profile!r}")
    return selected


def write_f16(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().view(np.uint16).tofile(path)


def read_f16(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != shape[0] * shape[1]:
        raise RuntimeError(f"bad output size: got {bits.size}, want {shape[0] * shape[1]}")
    return torch.from_numpy(bits.view(np.float16).reshape(shape).copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case, has_bias: bool) -> bool:
    seed = 3000 + case.m * 3 + case.k * 5 + case.n * 7 + (11 if has_bias else 0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(case.m, case.k, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    w = (torch.randn(case.k, case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    b = (torch.randn(case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    ref = x.to(torch.float32) @ w.to(torch.float32)
    if has_bias:
        ref = ref + b.to(torch.float32)
    ref = ref.to(torch.float16).to(torch.float32)

    suffix = "bias" if has_bias else "nobias"
    x_path = tmp / f"{case.name}.{suffix}.x.f16"
    w_path = tmp / f"{case.name}.{suffix}.w.f16"
    b_path = tmp / f"{case.name}.{suffix}.b.f16"
    y_path = tmp / f"{case.name}.{suffix}.y.f16"
    write_f16(x_path, x)
    write_f16(w_path, w)
    if has_bias:
        write_f16(b_path, b)
    env = gradients_env(root)
    proc = subprocess.run([str(runner), str(x_path), str(w_path), str(b_path) if has_bias else "-",
                           str(y_path), str(case.m), str(case.k), str(case.n)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"linear fwd actual skipped case={case.name} mode={suffix}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} mode={suffix} rc={proc.returncode}")
    actual = read_f16(y_path, (case.m, case.n))
    diff = (actual - ref).abs()
    rel = diff / torch.maximum(ref.abs(), torch.tensor(1.0e-6, dtype=torch.float32))
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    ok = bool(torch.all(diff <= (5.0e-3 + 5.0e-2 * ref.abs())).item())
    print(f"linear fwd actual {'ok' if ok else 'FAIL'} case={case.name} mode={suffix} "
          f"shape=({case.m},{case.k})x({case.k},{case.n}) max_abs={max_abs:.3e} "
          f"max_rel={max_rel:.3e} mean_abs={mean_abs:.3e}")
    return ok


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-linear-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp, "gd_linear_fwd_runner", RUNNER_SOURCE)
        failures = 0
        for case in make_cases():
            for has_bias in (False, True):
                if not run_case(runner, root, tmp, case, has_bias):
                    failures += 1
        if failures:
            raise SystemExit(f"linear fwd actual check failed cases={failures}")
    print("linear fwd actual checks ok")


if __name__ == "__main__":
    main()
