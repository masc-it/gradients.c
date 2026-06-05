# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c linear backward correctness check."""

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
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v == 0UL || v > UINT32_MAX) {
        return 1;
    }
    *out = (uint32_t)v;
    return 0;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX / b) { return 1; }
    *out = a * b;
    return 0;
}

static size_t align_up(size_t v, size_t a) { return (v + a - 1U) & ~(a - 1U); }

static int check_status(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) { return 0; }
    fprintf(stderr, "%s failed: %s (%d), ctx=%s\n", expr, gd_status_string(st), (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return 1;
}

#define CHECK(ctx, expr) do { if (check_status((ctx), (expr), #expr) != 0) { goto fail; } } while (0)

int main(int argc, char **argv)
{
    uint32_t m, k, n, has_bias;
    size_t x_elems, w_elems, g_elems, x_bytes, w_bytes, b_bytes, g_bytes, dx_bytes, dw_bytes;
    uint16_t *x_data = NULL, *w_data = NULL, *b_data = NULL, *g_data = NULL;
    uint16_t *dx_data = NULL, *dw_data = NULL, *db_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, b, g, dx, dw, db;
    int64_t x_shape[2], w_shape[2], b_shape[1], g_shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 11) {
        fprintf(stderr, "usage: %s X.bin W.bin B.bin|- G.bin DX.bin DW.bin DB.bin M K N\n", argv[0]);
        return 2;
    }
    if (parse_u32(argv[8], &m) != 0 || parse_u32(argv[9], &k) != 0 || parse_u32(argv[10], &n) != 0) {
        return 2;
    }
    has_bias = strcmp(argv[3], "-") != 0 ? 1U : 0U;
    if (mul_size((size_t)m, (size_t)k, &x_elems) != 0 || mul_size((size_t)k, (size_t)n, &w_elems) != 0 ||
        mul_size((size_t)m, (size_t)n, &g_elems) != 0 || mul_size(x_elems, 2U, &x_bytes) != 0 ||
        mul_size(w_elems, 2U, &w_bytes) != 0 || mul_size(g_elems, 2U, &g_bytes) != 0 ||
        mul_size((size_t)n, 2U, &b_bytes) != 0) { return 2; }
    dx_bytes = x_bytes;
    dw_bytes = w_bytes;

    x_data = (uint16_t *)malloc(x_bytes); w_data = (uint16_t *)malloc(w_bytes);
    b_data = (uint16_t *)malloc(b_bytes); g_data = (uint16_t *)malloc(g_bytes);
    dx_data = (uint16_t *)calloc(x_elems, sizeof(*dx_data));
    dw_data = (uint16_t *)calloc(w_elems, sizeof(*dw_data));
    db_data = (uint16_t *)calloc((size_t)n, sizeof(*db_data));
    if (x_data == NULL || w_data == NULL || b_data == NULL || g_data == NULL ||
        dx_data == NULL || dw_data == NULL || db_data == NULL) { goto fail; }
    if (read_file(argv[1], x_data, x_bytes) != 0 || read_file(argv[2], w_data, w_bytes) != 0 ||
        (has_bias != 0U && read_file(argv[3], b_data, b_bytes) != 0) ||
        read_file(argv[4], g_data, g_bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(x_bytes + w_bytes + b_bytes + g_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(dx_bytes + dw_bytes + b_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }

    x_shape[0] = (int64_t)m; x_shape[1] = (int64_t)k;
    w_shape[0] = (int64_t)k; w_shape[1] = (int64_t)n;
    b_shape[0] = (int64_t)n;
    g_shape[0] = (int64_t)m; g_shape[1] = (int64_t)n;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, g_shape), 256U, &g));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &w, w_data, w_bytes));
    CHECK(ctx, gd_tensor_write(ctx, &g, g_data, g_bytes));
    if (has_bias != 0U) {
        CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, b_shape), 256U, &b));
        CHECK(ctx, gd_tensor_write(ctx, &b, b_data, b_bytes));
    }
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    if (has_bias != 0U) {
        CHECK(ctx, gd_linear_backward(ctx, &x, &w, &b, &g, &dx, &dw, &db));
    } else {
        CHECK(ctx, gd_linear_backward(ctx, &x, &w, NULL, &g, &dx, &dw, NULL));
    }
    CHECK(ctx, gd_end_step(ctx)); CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, dx_bytes));
    CHECK(ctx, gd_tensor_read(ctx, &dw, dw_data, dw_bytes));
    if (has_bias != 0U) { CHECK(ctx, gd_tensor_read(ctx, &db, db_data, b_bytes)); }
    if (write_file(argv[5], dx_data, dx_bytes) != 0 || write_file(argv[6], dw_data, dw_bytes) != 0 ||
        (has_bias != 0U && write_file(argv[7], db_data, b_bytes) != 0)) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(w_data); free(b_data); free(g_data); free(dx_data); free(dw_data); free(db_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_linear_bwd_runner.c"
    binary = tmp / "gd_linear_bwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def make_cases() -> list[Case]:
    tokens = int(os.environ.get("GD_LINEAR_BWD_TOKENS", "512"))
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
    profile = os.environ.get("GD_LINEAR_BWD_PROFILE", "classic")
    if profile == "smoke":
        return smoke
    if profile in {"classic", "all"}:
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_LINEAR_BWD_PROFILE={profile!r}")
    return selected


def write_f16(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().view(np.uint16).tofile(path)


def read_f16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16)
    want = int(np.prod(shape))
    if bits.size != want:
        raise RuntimeError(f"bad output size: got {bits.size}, want {want}")
    return torch.from_numpy(bits.view(np.float16).reshape(shape).copy()).to(torch.float32)


def check_close(name: str, actual: torch.Tensor, ref: torch.Tensor) -> tuple[bool, str]:
    diff = (actual - ref).abs()
    rel = diff / torch.maximum(ref.abs(), torch.tensor(1.0e-6, dtype=torch.float32))
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    ok = bool(torch.all(diff <= (8.0e-3 + 7.0e-2 * ref.abs())).item())
    return ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} mean_abs={mean_abs:.3e}"


def run_case(runner: Path, root: Path, tmp: Path, case: Case, has_bias: bool) -> bool:
    seed = 4000 + case.m * 3 + case.k * 5 + case.n * 7 + (11 if has_bias else 0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(case.m, case.k, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    w = (torch.randn(case.k, case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    b = (torch.randn(case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    g = (torch.randn(case.m, case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    dx_ref = (g.to(torch.float32) @ w.to(torch.float32).T).to(torch.float16).to(torch.float32)
    dw_ref = (x.to(torch.float32).T @ g.to(torch.float32)).to(torch.float16).to(torch.float32)
    db_ref = g.to(torch.float32).sum(dim=0).to(torch.float16).to(torch.float32)

    suffix = "bias" if has_bias else "nobias"
    x_path = tmp / f"{case.name}.{suffix}.x.f16"
    w_path = tmp / f"{case.name}.{suffix}.w.f16"
    b_path = tmp / f"{case.name}.{suffix}.b.f16"
    g_path = tmp / f"{case.name}.{suffix}.g.f16"
    dx_path = tmp / f"{case.name}.{suffix}.dx.f16"
    dw_path = tmp / f"{case.name}.{suffix}.dw.f16"
    db_path = tmp / f"{case.name}.{suffix}.db.f16"
    for path, tensor in ((x_path, x), (w_path, w), (g_path, g)):
        write_f16(path, tensor)
    if has_bias:
        write_f16(b_path, b)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(x_path), str(w_path), str(b_path) if has_bias else "-",
                           str(g_path), str(dx_path), str(dw_path), str(db_path),
                           str(case.m), str(case.k), str(case.n)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"linear bwd actual skipped case={case.name} mode={suffix}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} mode={suffix} rc={proc.returncode}")
    dx_actual = read_f16(dx_path, (case.m, case.k))
    dw_actual = read_f16(dw_path, (case.k, case.n))
    dx_ok, dx_msg = check_close("dx", dx_actual, dx_ref)
    dw_ok, dw_msg = check_close("dw", dw_actual, dw_ref)
    ok = dx_ok and dw_ok
    db_msg = "db: skipped"
    if has_bias:
        db_actual = read_f16(db_path, (case.n,))
        db_ok, db_msg = check_close("db", db_actual, db_ref)
        ok = ok and db_ok
    print(f"linear bwd actual {'ok' if ok else 'FAIL'} case={case.name} mode={suffix} "
          f"shape=({case.m},{case.k})x({case.k},{case.n}) {dx_msg} {dw_msg} {db_msg}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-linear-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = 0
        for case in make_cases():
            for has_bias in (False, True):
                if not run_case(runner, root, tmp, case, has_bias):
                    failures += 1
        if failures:
            raise SystemExit(f"linear bwd actual check failed cases={failures}")
    print("linear bwd actual checks ok")


if __name__ == "__main__":
    main()
