# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c permute forward correctness check against PyTorch."""

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
    axes: tuple[int, ...]


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

static int parse_i64(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0') { return 1; }
    *out = (int64_t)v;
    return 0;
}

static int parse_i32(const char *s, int32_t *out)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v < INT32_MIN || v > INT32_MAX) { return 1; }
    *out = (int32_t)v;
    return 0;
}

static int parse_dtype(const char *s, gd_dtype *dtype, size_t *elem_size)
{
    if (strcmp(s, "f16") == 0) { *dtype = GD_DTYPE_F16; *elem_size = 2U; return 0; }
    if (strcmp(s, "f32") == 0) { *dtype = GD_DTYPE_F32; *elem_size = 4U; return 0; }
    if (strcmp(s, "u8") == 0) { *dtype = GD_DTYPE_U8; *elem_size = 1U; return 0; }
    if (strcmp(s, "i32") == 0) { *dtype = GD_DTYPE_I32; *elem_size = 4U; return 0; }
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
    size_t elem_size;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS] = {0};
    int32_t axes[GD_MAX_DIMS] = {0};
    size_t count = 1U;
    size_t bytes;
    unsigned char *x_host = NULL;
    unsigned char *y_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, y;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 5 || parse_dtype(argv[2], &dtype, &elem_size) != 0 ||
        parse_u32(argv[3], &rank) != 0 || rank > GD_MAX_DIMS || argc != (int)(5U + rank * 2U)) {
        fprintf(stderr, "usage: %s OUT.bin f16|f32|u8|i32 RANK DIMS... AXES... IN.bin\n", argv[0]);
        return 2;
    }
    arg = 4;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &shape[d]) != 0 || shape[d] <= 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)shape[d];
    }
    for (d = 0U; d < rank; ++d) {
        if (parse_i32(argv[arg++], &axes[d]) != 0) { return 2; }
    }
    if (count > SIZE_MAX / elem_size) { return 2; }
    bytes = count * elem_size;
    x_host = (unsigned char *)malloc(bytes);
    y_host = (unsigned char *)malloc(bytes);
    if (x_host == NULL || y_host == NULL || read_file(argv[arg], x_host, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 4U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), 256U, &x));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_host, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK(ctx, gd_permute(ctx, &x, axes, rank, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &y, y_host, bytes));
    if (write_file(argv[1], y_host, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_host);
    free(y_host);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_permute_fwd_runner.c"
    binary = tmp / "gd_permute_fwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a"), "-pthread", "-lm"]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def cases() -> list[Case]:
    all_cases = [
        Case("matrix_f32", "f32", (2, 3), (1, 0)),
        Case("qkv_f16", "f16", (2, 4, 3, 8), (0, 2, 1, 3)),
        Case("image_u8", "u8", (4, 5, 3), (2, 0, 1)),
        Case("tokens_i32", "i32", (3, 7), (1, 0)),
    ]
    profile = os.environ.get("GD_PERMUTE_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:3]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_PERMUTE_FWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "f32": torch.float32, "u8": torch.uint8, "i32": torch.int32}[dtype]


def np_dtype(dtype: str) -> np.dtype:
    return {"f16": np.float16, "f32": np.float32, "u8": np.uint8, "i32": np.int32}[dtype]


def make_input(case: Case) -> torch.Tensor:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x504D5445) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if case.dtype in {"f16", "f32"}:
        return (torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.25).to(torch_dtype(case.dtype))
    if case.dtype == "u8":
        return torch.randint(0, 251, case.shape, generator=gen, dtype=torch.uint8)
    return torch.randint(-1000, 1000, case.shape, generator=gen, dtype=torch.int32)


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x = make_input(case).contiguous()
    want = x.permute(*case.axes).contiguous()
    x_path = tmp / f"{case.name}_x.bin"
    y_path = tmp / f"{case.name}_y.bin"
    x_path.write_bytes(x.numpy().astype(np_dtype(case.dtype), copy=False).tobytes())
    cmd = [str(binary), str(y_path), case.dtype, str(len(case.shape)),
           *map(str, case.shape), *map(str, case.axes), str(x_path)]
    env = os.environ.copy()
    env.setdefault("GRADIENTS_METALLIB", str(root / "build" / "gradients.metallib"))
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(y_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(tuple(want.shape))
    np.testing.assert_array_equal(got, want.numpy())
    print(f"[permute/fwd] {case.name}: ok")


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
