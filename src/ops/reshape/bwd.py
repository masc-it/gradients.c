# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c reshape backward correctness check against PyTorch autograd."""

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
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


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
    size_t elem_size;
    uint32_t input_rank;
    uint32_t grad_rank;
    int64_t input_shape[GD_MAX_DIMS] = {0};
    int64_t grad_shape[GD_MAX_DIMS] = {0};
    size_t count = 1U;
    size_t grad_count = 1U;
    size_t bytes;
    unsigned char *grad_host = NULL;
    unsigned char *dx_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, grad_out, dx;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 7 || parse_dtype(argv[2], &dtype, &elem_size) != 0 ||
        parse_u32(argv[3], &input_rank) != 0 || parse_u32(argv[4], &grad_rank) != 0 ||
        input_rank > GD_MAX_DIMS || grad_rank > GD_MAX_DIMS ||
        argc != (int)(6U + input_rank + grad_rank)) {
        fprintf(stderr, "usage: %s DX.bin f16|f32 INPUT_RANK GRAD_RANK INPUT_DIMS... GRAD_DIMS... GRAD.bin\n", argv[0]);
        return 2;
    }
    arg = 5;
    for (d = 0U; d < input_rank; ++d) {
        if (parse_i64(argv[arg++], &input_shape[d]) != 0 || input_shape[d] <= 0 ||
            (uint64_t)input_shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)input_shape[d];
    }
    for (d = 0U; d < grad_rank; ++d) {
        if (parse_i64(argv[arg++], &grad_shape[d]) != 0 || grad_shape[d] <= 0 ||
            (uint64_t)grad_shape[d] > (uint64_t)(SIZE_MAX / grad_count)) { return 2; }
        grad_count *= (size_t)grad_shape[d];
    }
    if (count != grad_count || count > SIZE_MAX / elem_size) { return 2; }
    bytes = count * elem_size;
    grad_host = (unsigned char *)malloc(bytes);
    dx_host = (unsigned char *)malloc(bytes);
    if (grad_host == NULL || dx_host == NULL || read_file(argv[arg], grad_host, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_zeros(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(input_rank, input_shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(grad_rank, grad_shape), 256U, &grad_out));
    CHECK(ctx, gd_tensor_write(ctx, &grad_out, grad_host, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_reshape_backward(ctx, &x, &grad_out, &dx));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_host, bytes));
    if (write_file(argv[1], dx_host, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(grad_host);
    free(dx_host);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_reshape_bwd_runner.c"
    binary = tmp / "gd_reshape_bwd_runner"
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
        Case("matrix_f16", "f16", (2, 3), (3, 2)),
        Case("infer_rank3_f32", "f32", (2, 3, 4), (2, -1, 3)),
        Case("heads_f16", "f16", (2, 4, 8, 16), (2, 8, -1)),
    ]
    profile = os.environ.get("GD_RESHAPE_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RESHAPE_BWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if dtype == "f16" else torch.float32


def np_dtype(dtype: str) -> np.dtype:
    return np.float16 if dtype == "f16" else np.float32


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x9E5A) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.zeros(case.input_shape, dtype=torch_dtype(case.dtype), requires_grad=True)
    y = x.reshape(*case.target_shape)
    grad = (torch.randn(tuple(y.shape), generator=gen, dtype=torch.float32) * 0.2).to(torch_dtype(case.dtype))
    y.backward(grad)
    grad_path = tmp / f"{case.name}_grad.bin"
    dx_path = tmp / f"{case.name}_dx.bin"
    grad_path.write_bytes(grad.detach().contiguous().numpy().astype(np_dtype(case.dtype), copy=False).tobytes())
    cmd = [str(binary), str(dx_path), case.dtype, str(len(case.input_shape)), str(len(tuple(y.shape))),
           *map(str, case.input_shape), *map(str, tuple(y.shape)), str(grad_path)]
    env = os.environ.copy()
    env.setdefault("GRADIENTS_METALLIB", str(root / "build" / "gradients.metallib"))
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(dx_path.read_bytes(), dtype=np_dtype(case.dtype)).reshape(case.input_shape)
    want = x.grad.detach().numpy()
    np.testing.assert_array_equal(got, want)
    print(f"[reshape/bwd] {case.name}: ok")


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
