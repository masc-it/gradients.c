# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c sigmoid forward correctness check against PyTorch."""

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
    edge: bool = False


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
    unsigned char *x_data = NULL;
    unsigned char *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y;
    gd_status st;
    uint32_t i;
    int rc = 1;

    if (argc < 6 || parse_dtype(argv[3], &dtype, &elem_size) != 0 || parse_u32(argv[4], &rank) != 0 ||
        rank == 0U || rank > GD_MAX_DIMS || argc != (int)(5U + rank)) {
        fprintf(stderr, "usage: %s X.bin Y.bin f16|f32 RANK DIM...\n", argv[0]);
        return 2;
    }
    for (i = 0U; i < rank; ++i) {
        if (parse_i64_dim(argv[5U + i], &shape[i]) != 0 ||
            (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / elems)) {
            return 2;
        }
        elems *= (size_t)shape[i];
    }
    if (elems == 0U || elems > SIZE_MAX / elem_size) { return 2; }
    bytes = elems * elem_size;
    x_data = (unsigned char *)malloc(bytes);
    y_data = (unsigned char *)calloc(elems, elem_size);
    if (x_data == NULL || y_data == NULL || read_file(argv[1], x_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_sigmoid(ctx, &x, &y));
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_sigmoid_fwd_runner.c"
    binary = tmp / "gd_sigmoid_fwd_runner"
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
        Case("edge_f16", "f16", (1, 11), True),
        Case("tail_4x257_f16", "f16", (4, 257)),
        Case("small_f32", "f32", (7, 19)),
    ]
    classic = [
        Case("rank3_8x33x65_f16", "f16", (8, 33, 65)),
        Case("large_1024x1024_f16", "f16", (1024, 1024)),
        Case("rank3_3x17x64_f32", "f32", (3, 17, 64)),
    ]
    profile = os.environ.get("GD_SIGMOID_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_SIGMOID_FWD_PROFILE={profile!r}")
    return selected


def case_input(case: Case) -> torch.Tensor:
    dtype = torch.float16 if case.dtype == "f16" else torch.float32
    if case.edge:
        values = torch.tensor([-20.0, -12.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 12.0, 20.0], dtype=dtype)
        return values.reshape(case.shape).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(3000 + sum(case.shape) * 17 + len(case.shape))
    return (torch.randn(*case.shape, generator=gen, dtype=torch.float32) * 4.0).to(dtype).contiguous()


def tensor_bytes(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous().numpy()
    if dtype == "f16":
        return arr.astype(np.float16, copy=False).view(np.uint16)
    return arr.astype(np.float32, copy=False)


def read_actual(path: Path, case: Case) -> torch.Tensor:
    if case.dtype == "f16":
        arr = np.fromfile(path, dtype=np.uint16).reshape(case.shape).view(np.float16)
        return torch.from_numpy(arr.copy()).to(torch.float16)
    arr = np.fromfile(path, dtype=np.float32).reshape(case.shape)
    return torch.from_numpy(arr.copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x = case_input(case)
    ref = torch.sigmoid(x)
    x_path = tmp / f"{case.name}.x.bin"
    y_path = tmp / f"{case.name}.y.bin"
    tensor_bytes(x, case.dtype).tofile(x_path)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    cmd = [str(runner), str(x_path), str(y_path), case.dtype, str(len(case.shape)), *(str(d) for d in case.shape)]
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"sigmoid fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual = read_actual(y_path, case)
    tol = 1.2e-3 if case.dtype == "f16" else 6.0e-6
    max_abs = float((actual.float() - ref.float()).abs().max().item())
    ok = max_abs <= tol
    print(f"sigmoid fwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype={case.dtype} max_abs={max_abs:.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-sigmoid-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"sigmoid fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
