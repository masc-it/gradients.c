# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c ReLU forward correctness check against PyTorch."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    cols: int
    dtype: str


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v == 0UL || v > UINT32_MAX) { return 1; }
    *out = (uint32_t)v;
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

int main(int argc, char **argv)
{
    uint32_t rows, cols, dtype;
    size_t elems, elem_size, bytes;
    void *x_data = NULL;
    void *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y;
    int64_t shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 6 || parse_u32(argv[3], &rows) != 0 || parse_u32(argv[4], &cols) != 0 ||
        parse_u32(argv[5], &dtype) != 0) {
        fprintf(stderr, "usage: %s X.bin Y.bin ROWS COLS DTYPE\n", argv[0]);
        return 2;
    }
    elem_size = dtype == (uint32_t)GD_DTYPE_F32 ? 4U : 2U;
    elems = (size_t)rows * (size_t)cols;
    bytes = elems * elem_size;
    x_data = malloc(bytes);
    y_data = calloc(elems, elem_size);
    if (x_data == NULL || y_data == NULL || read_file(argv[1], x_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    shape[0] = (int64_t)rows; shape[1] = (int64_t)cols;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, (gd_dtype)dtype, 2U, shape, 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(ctx, gd_relu(ctx, &x, &y));
    CHECK(ctx, gd_end(ctx));
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_relu_fwd_runner.c"
    binary = tmp / "gd_relu_fwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def make_cases() -> list[Case]:
    smoke = [Case("small_f16", 4, 17, "f16"), Case("small_f32", 3, 19, "f32")]
    classic = [Case("large_f16", 1024, 1024, "f16"), Case("large_f32", 512, 512, "f32")]
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


def dtype_info(name: str) -> tuple[torch.dtype, np.dtype, int]:
    if name == "f16":
        return torch.float16, np.float16, 1
    if name == "f32":
        return torch.float32, np.float32, 3
    raise ValueError(name)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    torch_dtype, np_dtype, gd_dtype = dtype_info(case.dtype)
    gen = torch.Generator(device="cpu").manual_seed(1000 + case.rows * 3 + case.cols)
    x = (torch.randn(case.rows, case.cols, generator=gen, dtype=torch.float32) * 2.0 - 0.2).to(torch_dtype)
    ref = torch.relu(x).to(torch.float32)
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    x.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).tofile(x_path)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(x_path), str(y_path), str(case.rows), str(case.cols), str(gd_dtype)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"relu fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual_np = np.fromfile(y_path, dtype=np_dtype).reshape(case.rows, case.cols)
    actual = torch.from_numpy(actual_np.copy()).to(torch.float32)
    diff = (actual - ref).abs()
    max_abs = float(diff.max().item())
    ok = bool(torch.all(diff <= (1e-6 if case.dtype == "f32" else 0.0)).item())
    print(f"relu fwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} max_abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-relu-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"relu fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
