# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c reduce_mean forward correctness check against PyTorch."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AXIS_ALL (-999)

static int read_file(const char *path, void *dst, size_t nbytes)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    if (fread(dst, 1U, nbytes, f) != nbytes) { fclose(f); return 1; }
    return fclose(f) != 0 ? 1 : 0;
}

static int write_file(const char *path, const void *src, size_t nbytes)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    if (fwrite(src, 1U, nbytes, f) != nbytes) { fclose(f); return 1; }
    return fclose(f) != 0 ? 1 : 0;
}

static int parse_i32(const char *s, int32_t *out)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v < INT32_MIN || v > INT32_MAX) { return 1; }
    *out = (int32_t)v;
    return 0;
}

static int parse_i64(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v <= 0) { return 1; }
    *out = (int64_t)v;
    return 0;
}

static int check_status(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) { return 0; }
    fprintf(stderr, "%s failed: %s (%d), ctx=%s\n", expr, gd_status_string(st), (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return 1;
}

static size_t align_up(size_t v, size_t a) { return (v + a - 1U) & ~(a - 1U); }
#define CHECK(ctx, expr) do { if (check_status((ctx), (expr), #expr) != 0) { goto fail; } } while (0)

static int tensor_count(uint32_t rank, const int64_t *shape, size_t *out)
{
    uint32_t i;
    size_t count = 1U;
    for (i = 0U; i < rank; ++i) {
        if (shape[i] <= 0 || (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / count)) { return 1; }
        count *= (size_t)shape[i];
    }
    *out = count;
    return 0;
}

int main(int argc, char **argv)
{
    int32_t dtype_i, rank_i, axis;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    size_t x_count, out_count, elem_size, out_elem_size, x_bytes;
    void *x_data = NULL;
    void *out_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, out;
    gd_status st;
    int rc = 1;
    uint32_t i;

    if (argc < 7 || parse_i32(argv[3], &dtype_i) != 0 || parse_i32(argv[4], &rank_i) != 0 ||
        parse_i32(argv[5], &axis) != 0 || rank_i <= 0 || rank_i > (int32_t)GD_MAX_DIMS || argc != 6 + rank_i) {
        fprintf(stderr, "usage: %s X.bin OUT.bin DTYPE RANK AXIS DIMS...\n", argv[0]);
        return 2;
    }
    rank = (uint32_t)rank_i;
    for (i = 0U; i < rank; ++i) {
        if (parse_i64(argv[6 + i], &shape[i]) != 0) { return 2; }
    }
    dtype = (gd_dtype)dtype_i;
    elem_size = dtype == GD_DTYPE_F32 ? 4U : 2U;
    if (tensor_count(rank, shape, &x_count) != 0) { return 2; }
    x_bytes = x_count * elem_size;
    x_data = malloc(x_bytes);
    out_data = calloc(x_count, 4U);
    if (x_data == NULL || out_data == NULL || read_file(argv[1], x_data, x_bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(x_bytes + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(x_bytes * 4U + 32U * 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    if (axis == AXIS_ALL) {
        CHECK(ctx, gd_reduce_mean(ctx, &x, &out));
    } else {
        CHECK(ctx, gd_reduce_mean_axis(ctx, &x, axis, false, &out));
    }
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    if (tensor_count(out.rank, out.shape, &out_count) != 0) { goto fail; }
    out_elem_size = out.dtype == GD_DTYPE_F32 ? 4U : 2U;
    CHECK(ctx, gd_tensor_read(ctx, &out, out_data, out_count * out_elem_size));
    if (write_file(argv[2], out_data, out_count * out_elem_size) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(out_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_reduce_mean_fwd_runner.c"
    binary = tmp / "gd_reduce_mean_fwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


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
    profile = os.environ.get("GD_REDUCE_MEAN_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_REDUCE_MEAN_FWD_PROFILE={profile!r}")
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


def reference(x: torch.Tensor, case: Case) -> torch.Tensor:
    x32 = x.to(torch.float32)
    if case.axis == AXIS_ALL:
        return x32.mean()
    out = x32.mean(dim=case.axis)
    return out.to(x.dtype) if x.dtype == torch.float16 else out


def tensor_to_file(t: torch.Tensor, path: Path, np_dtype: np.dtype) -> None:
    t.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).tofile(path)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    torch_dtype, np_dtype, gd_dtype = dtype_info(case.dtype)
    gen = torch.Generator(device="cpu").manual_seed(5100 + sum(case.shape) * 7 + len(case.shape))
    x = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 0.75).to(torch_dtype)
    ref = reference(x, case)
    x_path = tmp / f"{case.name}.x.bin"
    out_path = tmp / f"{case.name}.out.bin"
    tensor_to_file(x, x_path, np_dtype)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(x_path), str(out_path), str(gd_dtype), str(len(case.shape)),
                           str(case.axis), *[str(dim) for dim in case.shape]],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"reduce_mean fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    out_np_dtype = np.float32 if case.axis == AXIS_ALL and case.dtype == "f16" else np_dtype
    actual = np.fromfile(out_path, dtype=out_np_dtype).reshape(output_shape(case))
    want = ref.detach().cpu().contiguous().numpy().astype(out_np_dtype, copy=False).reshape(output_shape(case))
    actual32 = actual.astype(np.float32, copy=False)
    want32 = want.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(actual32 - want32))) if actual32.size else float(abs(actual32.item() - want32.item()))
    tol = 2.0e-4 if case.axis == AXIS_ALL and case.dtype == "f16" else (2.0e-2 if case.dtype == "f16" else 2.0e-4)
    ok = max_abs <= tol
    print(f"reduce_mean fwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} "
          f"shape={case.shape} axis={case.axis} max_abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-reduce-mean-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"reduce_mean fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
