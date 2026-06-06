# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c rms_norm backward correctness check against PyTorch formula."""

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
    eps: float = 1.0e-5


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

static int parse_f32(const char *s, float *out)
{
    char *end = NULL;
    float v = strtof(s, &end);
    if (s == NULL || *s == '\0' || end == s || *end != '\0') { return 1; }
    *out = v;
    return 0;
}

static int parse_dtype(const char *s, gd_dtype *dtype)
{
    if (strcmp(s, "f16") == 0) { *dtype = GD_DTYPE_F16; return 0; }
    if (strcmp(s, "f32") == 0) { *dtype = GD_DTYPE_F32; return 0; }
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
    int64_t shape[GD_MAX_DIMS] = {0};
    int64_t w_shape[1];
    float eps;
    size_t count = 1U;
    size_t cols;
    float *x_host = NULL;
    float *w_host = NULL;
    float *g_host = NULL;
    float *dx_host = NULL;
    float *dw_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, g, dx, dw;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 9 || parse_dtype(argv[3], &dtype) != 0 || parse_f32(argv[4], &eps) != 0 ||
        parse_u32(argv[5], &rank) != 0 || rank == 0U || rank > GD_MAX_DIMS ||
        argc != (int)(9U + rank)) {
        fprintf(stderr, "usage: %s DX.f32 DW.f32 f16|f32 EPS RANK DIMS... X.f32 W.f32 G.f32\n", argv[0]);
        return 2;
    }
    arg = 6;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &shape[d]) != 0 || shape[d] <= 0 ||
            (uint64_t)shape[d] > (uint64_t)(SIZE_MAX / count)) { return 2; }
        count *= (size_t)shape[d];
    }
    cols = (size_t)shape[rank - 1U];
    w_shape[0] = (int64_t)cols;
    x_host = (float *)malloc(sizeof(float) * count);
    w_host = (float *)malloc(sizeof(float) * cols);
    g_host = (float *)malloc(sizeof(float) * count);
    dx_host = (float *)malloc(sizeof(float) * count);
    dw_host = (float *)malloc(sizeof(float) * cols);
    if (x_host == NULL || w_host == NULL || g_host == NULL || dx_host == NULL || dw_host == NULL ||
        read_file(argv[arg], x_host, sizeof(float) * count) != 0 ||
        read_file(argv[arg + 1], w_host, sizeof(float) * cols) != 0 ||
        read_file(argv[arg + 2], g_host, sizeof(float) * count) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(sizeof(float) * (count * 2U + cols) * 4U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(sizeof(float) * count * 24U + sizeof(float) * cols * 16U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), x_host, count, false, &x));
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(1U, w_shape), w_host, cols, false, &w));
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank, shape), g_host, count, false, &g));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_rms_norm_backward(ctx, &x, &w, &g, eps, &dx, &dw));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dx, dx_host, count));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dw, dw_host, cols));
    if (write_file(argv[1], dx_host, sizeof(float) * count) != 0 ||
        write_file(argv[2], dw_host, sizeof(float) * cols) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_host); free(w_host); free(g_host); free(dx_host); free(dw_host);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_rms_norm_bwd_runner.c"
    binary = tmp / "gd_rms_norm_bwd_runner"
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
        Case("tiny_f32", "f32", (2, 4)),
        Case("tokens_f16", "f16", (4, 17, 64)),
        Case("llama_hidden_f16", "f16", (2, 128, 4096)),
        Case("mlp_f32", "f32", (257, 1024)),
    ]
    profile = os.environ.get("GD_RMS_NORM_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RMS_NORM_BWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "f32": torch.float32}[dtype]


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x42574452) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.5
    w = torch.randn((case.shape[-1],), generator=gen, dtype=torch.float32) * 0.25 + 1.0
    g = torch.randn(case.shape, generator=gen, dtype=torch.float32) * 0.25
    return x, w, g


def reference(case: Case, x: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch_dtype(case.dtype)
    xq = x.to(dtype).float()
    wq = w.to(dtype).float()
    gq = g.to(dtype).float()
    cols = xq.shape[-1]
    inv = torch.rsqrt(xq.pow(2).mean(dim=-1, keepdim=True) + case.eps)
    a = (gq * wq * xq).sum(dim=-1, keepdim=True)
    dx = inv * gq * wq - xq * (inv ** 3) * a / float(cols)
    reduce_dims = tuple(range(xq.ndim - 1))
    dw = (gq * xq * inv).sum(dim=reduce_dims)
    if dtype == torch.float16:
        dx = dx.to(torch.float16).float()
        dw = dw.to(torch.float16).float()
    return dx.contiguous(), dw.contiguous()


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    x, w, g = make_inputs(case)
    want_dx, want_dw = reference(case, x, w, g)
    x_path = tmp / f"{case.name}_x.f32"
    w_path = tmp / f"{case.name}_w.f32"
    g_path = tmp / f"{case.name}_g.f32"
    dx_path = tmp / f"{case.name}_dx.f32"
    dw_path = tmp / f"{case.name}_dw.f32"
    x_path.write_bytes(x.contiguous().numpy().astype(np.float32).tobytes())
    w_path.write_bytes(w.contiguous().numpy().astype(np.float32).tobytes())
    g_path.write_bytes(g.contiguous().numpy().astype(np.float32).tobytes())
    cmd = [str(binary), str(dx_path), str(dw_path), case.dtype, str(case.eps), str(len(case.shape)),
           *map(str, case.shape), str(x_path), str(w_path), str(g_path)]
    env = os.environ.copy()
    env.setdefault("GRADIENTS_METALLIB", str(root / "build" / "gradients.metallib"))
    subprocess.run(cmd, check=True, env=env)
    got_dx = np.frombuffer(dx_path.read_bytes(), dtype=np.float32).reshape(case.shape)
    got_dw = np.frombuffer(dw_path.read_bytes(), dtype=np.float32).reshape((case.shape[-1],))
    atol = 5.0e-3 if case.dtype == "f16" else 3.0e-5
    rtol = 5.0e-3 if case.dtype == "f16" else 3.0e-5
    np.testing.assert_allclose(got_dx, want_dx.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(got_dw, want_dw.numpy(), atol=atol, rtol=rtol)
    print(f"[rms_norm/bwd] {case.name}: ok")


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
