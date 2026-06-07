# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c FP16 PoWLU forward check against PyTorch."""

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
    n: int
    m: float
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
    if (s == NULL || *s == '\0' || end == s || *end != '\0' || v == 0UL || v > UINT32_MAX) { return 1; }
    *out = (uint32_t)v;
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
    uint32_t n;
    float m;
    size_t bytes;
    float *x1_data = NULL;
    float *x2_data = NULL;
    float *y_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x1, x2, y;
    int64_t shape[1];
    gd_status st;
    int rc = 1;

    if (argc != 6 || parse_u32(argv[4], &n) != 0 || parse_f32(argv[5], &m) != 0) {
        fprintf(stderr, "usage: %s X1.f32 X2.f32 Y.f32 N M\n", argv[0]);
        return 2;
    }
    bytes = (size_t)n * sizeof(float);
    x1_data = (float *)malloc(bytes);
    x2_data = (float *)malloc(bytes);
    y_data = (float *)calloc((size_t)n, sizeof(float));
    if (x1_data == NULL || x2_data == NULL || y_data == NULL ||
        read_file(argv[1], x1_data, bytes) != 0 || read_file(argv[2], x2_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up((size_t)n * 2U * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up((size_t)n * 2U * 4U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    shape[0] = (int64_t)n;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x1));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x2));
    CHECK(ctx, gd_tensor_write_f32(ctx, &x1, x1_data, n));
    CHECK(ctx, gd_tensor_write_f32(ctx, &x2, x2_data, n));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_powlu(ctx, &x1, &x2, m, &y));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &y, y_data, n));
    if (write_file(argv[3], y_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x1_data); free(x2_data); free(y_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_powlu_fwd_runner.c"
    binary = tmp / "gd_powlu_fwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-lm", "-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def powlu(x1: torch.Tensor, x2: torch.Tensor, m: float) -> torch.Tensor:
    x1f = x1.to(torch.float32)
    z = x2.to(torch.float32)
    s = torch.sigmoid(z)
    pos = torch.pow(z.clamp_min(torch.finfo(torch.float32).tiny), m / (torch.sqrt(z.clamp_min(0.0)) + 1.0)) * s
    gate = torch.where(z <= 0.0, z * s, pos)
    return (x1f * gate).to(torch.float16).to(torch.float32)


def make_cases() -> list[Case]:
    smoke = [Case("edge", 17, 3.0, True), Case("tail", 513, 2.0)]
    full = smoke + [Case("gpt_block", 4096 * 4, 3.0)]
    profile = os.environ.get("GD_POWLU_FWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "all":
        return full
    selected = [case for case in full if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_POWLU_FWD_PROFILE={profile!r}")
    return selected


def case_input(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    if case.edge:
        x1 = torch.tensor([-1.5, -0.75, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
                           1.5, -1.0, 2.0, -2.0, 0.375, -0.625, 1.75, -1.25], dtype=torch.float32)
        x2 = torch.tensor([-4.0, -2.0, -1.0, -0.5, -0.125, 0.0, 0.125, 0.25, 0.5,
                           1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 0.75], dtype=torch.float32)
        return x1, x2
    gen = torch.Generator(device="cpu").manual_seed(1234 + case.n)
    x1 = torch.randn(case.n, generator=gen, dtype=torch.float32) * 1.5
    x2 = torch.randn(case.n, generator=gen, dtype=torch.float32) * 2.0
    return x1, x2


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x1, x2 = case_input(case)
    x1q = x1.to(torch.float16)
    x2q = x2.to(torch.float16)
    ref = powlu(x1q, x2q, case.m)
    x1_path = tmp / f"{case.name}.x1.f32"
    x2_path = tmp / f"{case.name}.x2.f32"
    y_path = tmp / f"{case.name}.y.f32"
    x1.cpu().numpy().astype(np.float32).tofile(x1_path)
    x2.cpu().numpy().astype(np.float32).tofile(x2_path)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(x1_path), str(x2_path), str(y_path), str(case.n), str(case.m)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"powlu fwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual = torch.from_numpy(np.fromfile(y_path, dtype=np.float32))
    diff = (actual - ref).abs()
    ok = bool(torch.all(diff <= 2.5e-3))
    print(f"powlu fwd actual {'ok' if ok else 'FAIL'} case={case.name} n={case.n} m={case.m:g} max_abs={float(diff.max()):.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-powlu-fwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"powlu fwd check failed cases={failures}")


if __name__ == "__main__":
    main()
