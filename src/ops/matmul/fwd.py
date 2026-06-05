# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Matmul forward reference + live gradients.c correctness check.

This script keeps the PyTorch reference check, then builds the library and runs
actual `gd_matmul` calls through a tiny C harness.  The harness writes the Metal
result bytes back to Python so we can compare the production metallib path
against PyTorch for classic transformer-training GEMM shapes.

Run from the repository root with:

    uv run src/ops/matmul/fwd.py

Useful knobs:

    GD_MATMUL_FWD_PROFILE=smoke|classic|all   # default: classic
    GD_MATMUL_FWD_TOKENS=512                  # default sequence/tokens
"""

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
    m: int
    k: int
    n: int


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_file(const char *path, void *dst, size_t nbytes)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "failed to open input %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fread(dst, 1U, nbytes, f) != nbytes) {
        fprintf(stderr, "failed to read %zu bytes from %s\n", nbytes, path);
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        return 1;
    }
    return 0;
}

static int write_file(const char *path, const void *src, size_t nbytes)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "failed to open output %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fwrite(src, 1U, nbytes, f) != nbytes) {
        fprintf(stderr, "failed to write %zu bytes to %s\n", nbytes, path);
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        return 1;
    }
    return 0;
}

static int parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v;
    if (s == NULL || out == NULL || s[0] == '\0') {
        return 1;
    }
    v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v == 0UL || v > (unsigned long)UINT32_MAX) {
        return 1;
    }
    *out = (uint32_t)v;
    return 0;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static int check_status(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return 0;
    }
    fprintf(stderr, "%s failed: %s (%d), ctx=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return 1;
}

#define CHECK(ctx, expr) do { if (check_status((ctx), (expr), #expr) != 0) { goto fail; } } while (0)

int main(int argc, char **argv)
{
    uint32_t m;
    uint32_t k;
    uint32_t n;
    size_t x_elems;
    size_t w_elems;
    size_t y_elems;
    size_t x_bytes;
    size_t w_bytes;
    size_t y_bytes;
    uint16_t *x_data = NULL;
    uint16_t *w_data = NULL;
    uint16_t *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x;
    gd_tensor w;
    gd_tensor y;
    int64_t x_shape[2];
    int64_t w_shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 7) {
        fprintf(stderr, "usage: %s X.bin W.bin Y.bin M K N\n", argv[0]);
        return 2;
    }
    if (parse_u32(argv[4], &m) != 0 || parse_u32(argv[5], &k) != 0 ||
        parse_u32(argv[6], &n) != 0) {
        fprintf(stderr, "invalid shape arguments\n");
        return 2;
    }
    if (mul_size((size_t)m, (size_t)k, &x_elems) != 0 ||
        mul_size((size_t)k, (size_t)n, &w_elems) != 0 ||
        mul_size((size_t)m, (size_t)n, &y_elems) != 0 ||
        mul_size(x_elems, 2U, &x_bytes) != 0 ||
        mul_size(w_elems, 2U, &w_bytes) != 0 ||
        mul_size(y_elems, 2U, &y_bytes) != 0) {
        fprintf(stderr, "shape byte size overflow\n");
        return 2;
    }

    x_data = (uint16_t *)malloc(x_bytes);
    w_data = (uint16_t *)malloc(w_bytes);
    y_data = (uint16_t *)calloc(y_elems, sizeof(*y_data));
    if (x_data == NULL || w_data == NULL || y_data == NULL) {
        fprintf(stderr, "host allocation failed\n");
        goto fail;
    }
    if (read_file(argv[1], x_data, x_bytes) != 0 || read_file(argv[2], w_data, w_bytes) != 0) {
        goto fail;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(x_bytes + w_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(y_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("SKIP unsupported backend\n");
        rc = 77;
        goto done;
    }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) {
        goto fail;
    }

    x_shape[0] = (int64_t)m;
    x_shape[1] = (int64_t)k;
    w_shape[0] = (int64_t)k;
    w_shape[1] = (int64_t)n;

    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &w, w_data, w_bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_matmul(ctx, &x, &w, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &y, y_data, y_bytes));

    if (write_file(argv[3], y_data, y_bytes) != 0) {
        goto fail;
    }
    rc = 0;
    goto done;

fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data);
    free(w_data);
    free(y_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_matmul_fwd_runner.c"
    binary = tmp / "gd_matmul_fwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = [
        "cc",
        f"-I{root / 'include'}",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source),
        str(root / "build" / "libgradients.a"),
    ]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def make_cases() -> list[Case]:
    tokens = int(os.environ.get("GD_MATMUL_FWD_TOKENS", "512"))
    smoke = [Case("fallback_small", 4, 7, 6)]
    classic: list[Case] = []
    for hidden in (128, 256, 512):
        heads = 4
        head_dim = hidden // heads
        intermediate = hidden * 4
        classic.extend(
            [
                Case(f"h{hidden}_qkv", tokens, hidden, 3 * hidden),
                Case(f"h{hidden}_proj", tokens, hidden, hidden),
                Case(f"h{hidden}_mlp_up", tokens, hidden, intermediate),
                Case(f"h{hidden}_mlp_down", tokens, intermediate, hidden),
                Case(f"h{hidden}_attn_scores_h0", tokens, head_dim, tokens),
                Case(f"h{hidden}_attn_values_h0", tokens, tokens, head_dim),
            ]
        )
    profile = os.environ.get("GD_MATMUL_FWD_PROFILE", "classic")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return smoke + classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_MATMUL_FWD_PROFILE={profile!r}")
    return selected


def write_f16(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().contiguous().numpy().view(np.uint16)
    arr.tofile(path)


def read_f16(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != shape[0] * shape[1]:
        raise RuntimeError(f"bad output size: got {bits.size}, want {shape[0] * shape[1]}")
    values = bits.view(np.float16).reshape(shape)
    return torch.from_numpy(values.copy()).to(torch.float32)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    seed = 1000 + case.m * 3 + case.k * 5 + case.n * 7
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x_f16 = (torch.randn(case.m, case.k, generator=gen, dtype=torch.float32) * 0.05).to(torch.float16)
    w_f16 = (torch.randn(case.k, case.n, generator=gen, dtype=torch.float32) * 0.05).to(torch.float16)
    y_ref = (x_f16.to(torch.float32) @ w_f16.to(torch.float32)).to(torch.float16).to(torch.float32)

    x_path = tmp / f"{case.name}.x.f16"
    w_path = tmp / f"{case.name}.w.f16"
    y_path = tmp / f"{case.name}.y.f16"
    write_f16(x_path, x_f16)
    write_f16(w_path, w_f16)

    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run(
        [str(runner), str(x_path), str(w_path), str(y_path), str(case.m), str(case.k), str(case.n)],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 77:
        print(f"matmul fwd actual skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        raise RuntimeError(f"runner failed for case={case.name} rc={proc.returncode}")

    y_actual = read_f16(y_path, (case.m, case.n))
    diff = (y_actual - y_ref).abs()
    denom = torch.maximum(y_ref.abs(), torch.tensor(1.0e-6, dtype=torch.float32))
    rel = diff / denom
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    atol = 5.0e-3
    rtol = 5.0e-2
    ok = bool(torch.all(diff <= (atol + rtol * y_ref.abs())).item())
    status = "ok" if ok else "FAIL"
    print(
        f"matmul fwd actual {status} case={case.name} "
        f"shape=({case.m},{case.k})x({case.k},{case.n}) "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} mean_abs={mean_abs:.3e}"
    )
    return ok


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 7, dtype=torch.float32)
    w = torch.randn(7, 6, dtype=torch.float32)
    y_ref = x @ w
    y_loop = torch.zeros_like(y_ref)
    for i in range(x.shape[0]):
        for j in range(w.shape[1]):
            acc = torch.tensor(0.0, dtype=torch.float32)
            for k in range(x.shape[1]):
                acc = acc + x[i, k] * w[k, j]
            y_loop[i, j] = acc
    max_abs = (y_ref - y_loop).abs().max().item()
    assert max_abs < 1e-6, max_abs
    print(f"matmul fwd reference ok max_abs={max_abs:.3e} shape={tuple(y_ref.shape)}")

    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-matmul-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = 0
        for case in make_cases():
            if not run_case(runner, root, tmp, case):
                failures += 1
        if failures:
            raise SystemExit(f"matmul fwd actual check failed cases={failures}")
    print("matmul fwd actual checks ok")


if __name__ == "__main__":
    main()
