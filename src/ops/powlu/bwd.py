# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c FP16 PoWLU backward check against PyTorch/reference math."""

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
    float *g_data = NULL;
    float *dx1_data = NULL;
    float *dx2_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x1, x2, grad, dx1, dx2;
    int64_t shape[1];
    gd_status st;
    int rc = 1;

    if (argc != 8 || parse_u32(argv[6], &n) != 0 || parse_f32(argv[7], &m) != 0) {
        fprintf(stderr, "usage: %s X1.f32 X2.f32 G.f32 DX1.f32 DX2.f32 N M\n", argv[0]);
        return 2;
    }
    bytes = (size_t)n * sizeof(float);
    x1_data = (float *)malloc(bytes);
    x2_data = (float *)malloc(bytes);
    g_data = (float *)malloc(bytes);
    dx1_data = (float *)calloc((size_t)n, sizeof(float));
    dx2_data = (float *)calloc((size_t)n, sizeof(float));
    if (x1_data == NULL || x2_data == NULL || g_data == NULL || dx1_data == NULL || dx2_data == NULL ||
        read_file(argv[1], x1_data, bytes) != 0 || read_file(argv[2], x2_data, bytes) != 0 ||
        read_file(argv[3], g_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up((size_t)n * 2U * 3U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up((size_t)n * 2U * 8U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    shape[0] = (int64_t)n;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x1));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x2));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad));
    CHECK(ctx, gd_tensor_write_f32(ctx, &x1, x1_data, n));
    CHECK(ctx, gd_tensor_write_f32(ctx, &x2, x2_data, n));
    CHECK(ctx, gd_tensor_write_f32(ctx, &grad, g_data, n));
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_powlu_backward(ctx, &x1, &x2, &grad, m, &dx1, &dx2));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dx1, dx1_data, n));
    CHECK(ctx, gd_tensor_read_f32(ctx, &dx2, dx2_data, n));
    if (write_file(argv[4], dx1_data, bytes) != 0 || write_file(argv[5], dx2_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x1_data); free(x2_data); free(g_data); free(dx1_data); free(dx2_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_powlu_bwd_runner.c"
    binary = tmp / "gd_powlu_bwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-lm", "-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def gate_and_grad(z_in: torch.Tensor, m: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = z_in.to(torch.float32)
    s = torch.sigmoid(z)
    neg_gate = z * s
    neg_grad = s * (1.0 + z * (1.0 - s))
    zpos = z.clamp_min(torch.finfo(torch.float32).tiny)
    r = torch.sqrt(z.clamp_min(0.0))
    rp1 = r + 1.0
    a = m / rp1
    lz = torch.log(zpos)
    g = torch.exp(a * lz)
    da = -m / (2.0 * r.clamp_min(torch.finfo(torch.float32).tiny) * rp1 * rp1)
    pos_gate = g * s
    pos_grad = g * s * (a / zpos + da * lz + (1.0 - s))
    return torch.where(z <= 0.0, neg_gate, pos_gate), torch.where(z <= 0.0, neg_grad, pos_grad)


def ref_backward(x1: torch.Tensor, x2: torch.Tensor, grad: torch.Tensor, m: float) -> tuple[torch.Tensor, torch.Tensor]:
    gate, gate_grad = gate_and_grad(x2, m)
    dx1 = (grad.to(torch.float32) * gate).to(torch.float16).to(torch.float32)
    dx2 = (grad.to(torch.float32) * x1.to(torch.float32) * gate_grad).to(torch.float16).to(torch.float32)
    return dx1, dx2


def make_cases() -> list[Case]:
    smoke = [Case("edge", 17, 3.0, True), Case("tail", 513, 2.0)]
    full = smoke + [Case("gpt_block", 4096 * 4, 3.0)]
    profile = os.environ.get("GD_POWLU_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "all":
        return full
    selected = [case for case in full if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_POWLU_BWD_PROFILE={profile!r}")
    return selected


def case_input(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if case.edge:
        x1 = torch.tensor([-1.5, -0.75, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
                           1.5, -1.0, 2.0, -2.0, 0.375, -0.625, 1.75, -1.25], dtype=torch.float32)
        x2 = torch.tensor([-4.0, -2.0, -1.0, -0.5, -0.125, 0.0, 0.125, 0.25, 0.5,
                           1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 0.75], dtype=torch.float32)
        grad = torch.tensor([0.25, -0.375, 0.5, -0.625, 0.75, -0.875, 1.0, -1.125, 1.25,
                             -1.375, 1.5, -1.625, 1.75, -1.875, 2.0, -2.125, 2.25], dtype=torch.float32)
        return x1, x2, grad
    gen = torch.Generator(device="cpu").manual_seed(4321 + case.n)
    x1 = torch.randn(case.n, generator=gen, dtype=torch.float32) * 1.5
    x2 = torch.randn(case.n, generator=gen, dtype=torch.float32) * 2.0
    grad = torch.randn(case.n, generator=gen, dtype=torch.float32) * 0.75
    return x1, x2, grad


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x1, x2, grad = case_input(case)
    x1q = x1.to(torch.float16)
    x2q = x2.to(torch.float16)
    gradq = grad.to(torch.float16)
    ref_dx1, ref_dx2 = ref_backward(x1q, x2q, gradq, case.m)
    paths = {name: tmp / f"{case.name}.{name}.f32" for name in ["x1", "x2", "g", "dx1", "dx2"]}
    x1.cpu().numpy().astype(np.float32).tofile(paths["x1"])
    x2.cpu().numpy().astype(np.float32).tofile(paths["x2"])
    grad.cpu().numpy().astype(np.float32).tofile(paths["g"])
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(paths["x1"]), str(paths["x2"]), str(paths["g"]),
                           str(paths["dx1"]), str(paths["dx2"]), str(case.n), str(case.m)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"powlu bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual_dx1 = torch.from_numpy(np.fromfile(paths["dx1"], dtype=np.float32))
    actual_dx2 = torch.from_numpy(np.fromfile(paths["dx2"], dtype=np.float32))
    diff1 = (actual_dx1 - ref_dx1).abs()
    diff2 = (actual_dx2 - ref_dx2).abs()
    ok = bool(torch.all(diff1 <= 3.0e-3) and torch.all(diff2 <= 6.0e-3))
    print(f"powlu bwd actual {'ok' if ok else 'FAIL'} case={case.name} n={case.n} m={case.m:g} "
          f"dx1_max_abs={float(diff1.max()):.3e} dx2_max_abs={float(diff2.max()):.3e}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-powlu-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"powlu bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
