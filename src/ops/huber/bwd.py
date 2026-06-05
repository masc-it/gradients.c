# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c Huber backward correctness check against PyTorch."""

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
    dtype: str
    shape: tuple[int, ...]
    grad_out: float


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
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v > UINT32_MAX) { return 1; }
    *out = (uint32_t)v;
    return 0;
}

static int parse_i64_dim(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v <= 0) { return 1; }
    *out = (int64_t)v;
    return 0;
}

static int parse_f32(const char *s, float *out)
{
    char *end = NULL;
    float v = strtof(s, &end);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v != v) { return 1; }
    *out = v;
    return 0;
}

static int parse_dtype(const char *s, gd_dtype *dtype, size_t *elem_size)
{
    if (strcmp(s, "f16") == 0) { *dtype = GD_DTYPE_F16; *elem_size = 2U; return 0; }
    if (strcmp(s, "f32") == 0) { *dtype = GD_DTYPE_F32; *elem_size = 4U; return 0; }
    return 1;
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
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    size_t elem_size;
    size_t elems = 1U;
    size_t bytes;
    float grad_value;
    unsigned char *x_data = NULL;
    unsigned char *y_data = NULL;
    unsigned char *dx_data = NULL;
    unsigned char *dy_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y, grad, dx, dy;
    gd_status st;
    uint32_t i;
    int rc = 1;

    if (argc < 9 || parse_dtype(argv[5], &dtype, &elem_size) != 0 || parse_f32(argv[6], &grad_value) != 0 ||
        parse_u32(argv[7], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS || argc != (int)(8U + rank)) {
        fprintf(stderr, "usage: %s X.bin Y.bin DX.bin DY.bin f16|f32 grad RANK DIM...\n", argv[0]);
        return 2;
    }
    for (i = 0U; i < rank; ++i) {
        if (parse_i64_dim(argv[8U + i], &shape[i]) != 0 ||
            (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / elems)) { return 2; }
        elems *= (size_t)shape[i];
    }
    if (elems == 0U || elems > SIZE_MAX / elem_size) { return 2; }
    bytes = elems * elem_size;
    x_data = (unsigned char *)malloc(bytes);
    y_data = (unsigned char *)malloc(bytes);
    dx_data = (unsigned char *)calloc(elems, elem_size);
    dy_data = (unsigned char *)calloc(elems, elem_size);
    if (x_data == NULL || y_data == NULL || dx_data == NULL || dy_data == NULL ||
        read_file(argv[1], x_data, bytes) != 0 || read_file(argv[2], y_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 4U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &y));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(0U, NULL), 256U, &grad));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &y, y_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &grad, &grad_value, sizeof(grad_value)));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_huber_backward(ctx, &x, &y, &grad, &dx, &dy));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, bytes));
    CHECK(ctx, gd_tensor_read(ctx, &dy, dy_data, bytes));
    if (write_file(argv[3], dx_data, bytes) != 0 || write_file(argv[4], dy_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(y_data); free(dx_data); free(dy_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_huber_bwd_runner.c"
    binary = tmp / "gd_huber_bwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a"), "-pthread"]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def make_cases() -> list[Case]:
    smoke = [
        Case("small_f16", "f16", (4, 7), 0.75),
        Case("tail_f16", "f16", (4099,), -1.25),
        Case("small_f32", "f32", (5, 13), 1.5),
    ]
    classic = [
        Case("activation_1024x1024_f16", "f16", (1024, 1024), 1.0),
        Case("segmentation_8x128x128_f16", "f16", (8, 128, 128), -0.5),
        Case("regression_64x1024_f32", "f32", (64, 1024), 2.0),
    ]
    profile = os.environ.get("GD_HUBER_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_HUBER_BWD_PROFILE={profile!r}")
    return selected


def case_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if case.dtype == "f16" else torch.float32
    gen = torch.Generator(device="cpu").manual_seed(8000 + len(case.shape) * 131 + sum(case.shape))
    x = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    y = (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 1.75).to(dtype).contiguous()
    return x, y


def tensor_bytes(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous().numpy()
    if dtype == "f16":
        return arr.astype(np.float16, copy=False).view(np.uint16)
    return arr.astype(np.float32, copy=False)


def read_tensor(path: Path, case: Case) -> torch.Tensor:
    if case.dtype == "f16":
        arr = np.fromfile(path, dtype=np.uint16).reshape(case.shape).view(np.float16)
        return torch.from_numpy(arr.copy()).to(torch.float16)
    arr = np.fromfile(path, dtype=np.float32).reshape(case.shape)
    return torch.from_numpy(arr.copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x, y = case_inputs(case)
    diff = x.float() - y.float()
    clipped = torch.clamp(diff, min=-1.0, max=1.0)
    ref_dx = clipped * (case.grad_out / x.numel())
    ref_dy = -ref_dx
    if case.dtype == "f16":
        ref_dx = ref_dx.to(torch.float16).float()
        ref_dy = ref_dy.to(torch.float16).float()
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    dx_path = tmp / f"{case.name}.dx.bin"
    dy_path = tmp / f"{case.name}.dy.bin"
    tensor_bytes(x, case.dtype).tofile(x_path)
    tensor_bytes(y, case.dtype).tofile(y_path)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    cmd = [str(runner), str(x_path), str(y_path), str(dx_path), str(dy_path),
           case.dtype, str(case.grad_out), str(len(case.shape)), *(str(d) for d in case.shape)]
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"huber bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    dx = read_tensor(dx_path, case).float()
    dy = read_tensor(dy_path, case).float()
    tol = 2.0e-3 if case.dtype == "f16" else 5.0e-6
    max_dx = float((dx - ref_dx).abs().max().item())
    max_dy = float((dy - ref_dy).abs().max().item())
    ok = max(max_dx, max_dy) <= tol
    print(f"huber bwd {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} dx={max_dx:.3e} dy={max_dy:.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-huber-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"huber bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
