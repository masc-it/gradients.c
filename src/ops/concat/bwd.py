# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c concat backward correctness check against PyTorch autograd."""

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
    axis: int
    shapes: tuple[tuple[int, ...], ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS 8U
#define PATH_MAX_LOCAL 4096U

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
    size_t elem_size;
    int32_t axis;
    uint32_t rank;
    uint32_t n_inputs;
    int64_t shapes[MAX_INPUTS][GD_MAX_DIMS];
    int64_t out_shape[GD_MAX_DIMS];
    size_t counts[MAX_INPUTS];
    size_t bytes[MAX_INPUTS];
    size_t out_count = 1U;
    size_t out_bytes;
    unsigned char *grad_host = NULL;
    unsigned char *tmp_host = NULL;
    gd_tensor tensors[MAX_INPUTS];
    gd_tensor grads[MAX_INPUTS];
    const gd_tensor *inputs[MAX_INPUTS];
    gd_tensor grad_out;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    uint32_t i, d;
    int32_t norm_axis;
    int arg;
    int rc = 1;

    if (argc < 8 || parse_dtype(argv[2], &dtype, &elem_size) != 0 ||
        parse_i32(argv[3], &axis) != 0 || parse_u32(argv[4], &rank) != 0 ||
        parse_u32(argv[5], &n_inputs) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        n_inputs == 0U || n_inputs > MAX_INPUTS || argc != (int)(7U + n_inputs * rank)) {
        fprintf(stderr, "usage: %s OUT_PREFIX f16|f32 AXIS RANK N_INPUTS DIMS... GRAD.bin\n", argv[0]);
        return 2;
    }
    norm_axis = axis < 0 ? axis + (int32_t)rank : axis;
    if (norm_axis < 0 || norm_axis >= (int32_t)rank) { return 2; }
    arg = 6;
    for (i = 0U; i < n_inputs; ++i) {
        counts[i] = 1U;
        for (d = 0U; d < rank; ++d) {
            if (parse_i64_dim(argv[arg++], &shapes[i][d]) != 0 ||
                (uint64_t)shapes[i][d] > (uint64_t)(SIZE_MAX / counts[i])) { return 2; }
            counts[i] *= (size_t)shapes[i][d];
        }
        if (counts[i] > SIZE_MAX / elem_size) { return 2; }
        bytes[i] = counts[i] * elem_size;
    }
    for (d = 0U; d < rank; ++d) { out_shape[d] = shapes[0][d]; }
    out_shape[norm_axis] = 0;
    for (i = 0U; i < n_inputs; ++i) {
        for (d = 0U; d < rank; ++d) {
            if (d != (uint32_t)norm_axis && shapes[i][d] != shapes[0][d]) { return 2; }
        }
        out_shape[norm_axis] += shapes[i][norm_axis];
    }
    for (d = 0U; d < rank; ++d) {
        if ((uint64_t)out_shape[d] > (uint64_t)(SIZE_MAX / out_count)) { return 2; }
        out_count *= (size_t)out_shape[d];
    }
    if (out_count > SIZE_MAX / elem_size) { return 2; }
    out_bytes = out_count * elem_size;
    grad_host = (unsigned char *)malloc(out_bytes);
    tmp_host = (unsigned char *)malloc(out_bytes);
    if (grad_host == NULL || tmp_host == NULL || read_file(argv[arg], grad_host, out_bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(out_bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(out_bytes * 3U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    for (i = 0U; i < n_inputs; ++i) {
        CHECK(ctx, gd_tensor_zeros(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shapes[i]), 256U, &tensors[i]));
        inputs[i] = &tensors[i];
    }
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, out_shape), 256U, &grad_out));
    CHECK(ctx, gd_tensor_write(ctx, &grad_out, grad_host, out_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_concat_backward(ctx, &grad_out, inputs, n_inputs, axis, grads));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    for (i = 0U; i < n_inputs; ++i) {
        char path[PATH_MAX_LOCAL];
        int n = snprintf(path, sizeof(path), "%s_%u.bin", argv[1], i);
        if (n < 0 || (size_t)n >= sizeof(path)) { goto fail; }
        CHECK(ctx, gd_tensor_read(ctx, &grads[i], tmp_host, bytes[i]));
        if (write_file(path, tmp_host, bytes[i]) != 0) { goto fail; }
    }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(grad_host);
    free(tmp_host);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_concat_bwd_runner.c"
    binary = tmp / "gd_concat_bwd_runner"
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
        Case("axis0_f16", "f16", 0, ((3, 5), (2, 5))),
        Case("axis1_f16", "f16", 1, ((4, 7), (4, 3), (4, 5))),
        Case("rank3_neg_axis_f32", "f32", -1, ((2, 3, 4), (2, 3, 5))),
        Case("features_f32", "f32", 1, ((128, 256), (128, 512))),
    ]
    profile = os.environ.get("GD_CONCAT_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:3]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_CONCAT_BWD_PROFILE={profile!r}")
    return selected


def np_dtype(dtype: str) -> np.dtype:
    return np.float16 if dtype == "f16" else np.float32


def run_case(binary: Path, tmp: Path, case: Case) -> None:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0xBADC0DE) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    xs = [torch.zeros(shape, dtype=torch.float16 if case.dtype == "f16" else torch.float32,
                      requires_grad=True) for shape in case.shapes]
    y = torch.cat(xs, dim=case.axis)
    grad = torch.from_numpy(rng.normal(0.0, 0.2, tuple(y.shape)).astype(np_dtype(case.dtype)))
    y.backward(grad)
    grad_path = tmp / f"{case.name}_grad.bin"
    grad_path.write_bytes(grad.numpy().tobytes())
    prefix = tmp / f"{case.name}_dx"
    dims = [str(dim) for shape in case.shapes for dim in shape]
    cmd = [str(binary), str(prefix), case.dtype, str(case.axis), str(len(case.shapes[0])),
           str(len(case.shapes)), *dims, str(grad_path)]
    env = os.environ.copy()
    env.setdefault("GRADIENTS_METALLIB", str(repo_root() / "build" / "gradients.metallib"))
    subprocess.run(cmd, check=True, env=env)
    for i, x in enumerate(xs):
        got = np.frombuffer((tmp / f"{case.name}_dx_{i}.bin").read_bytes(), dtype=np_dtype(case.dtype)).reshape(tuple(x.shape))
        want = x.grad.detach().numpy()
        np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=0, atol=0)
    print(f"[concat/bwd] {case.name}: ok")


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp)
        for case in cases():
            run_case(binary, tmp, case)


if __name__ == "__main__":
    main()
