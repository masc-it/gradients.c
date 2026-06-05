# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c FP16 ReLU autograd/backward correctness check against PyTorch."""

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
    rows: int
    cols: int
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
    uint32_t rows, cols;
    size_t elems, bytes;
    uint16_t *x_data = NULL;
    uint16_t *g_data = NULL;
    uint16_t *dx_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, g, y, dx;
    int64_t shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 6 || parse_u32(argv[4], &rows) != 0 || parse_u32(argv[5], &cols) != 0) {
        fprintf(stderr, "usage: %s X.bin G.bin DX.bin ROWS COLS\n", argv[0]);
        return 2;
    }
    elems = (size_t)rows * (size_t)cols;
    bytes = elems * sizeof(uint16_t);
    x_data = (uint16_t *)malloc(bytes);
    g_data = (uint16_t *)malloc(bytes);
    dx_data = (uint16_t *)calloc(elems, sizeof(uint16_t));
    if (x_data == NULL || g_data == NULL || dx_data == NULL ||
        read_file(argv[1], x_data, bytes) != 0 || read_file(argv[2], g_data, bytes) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 4U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U; cfg.data_slots = 2U; cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }
    shape[0] = (int64_t)rows; shape[1] = (int64_t)cols;
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &g));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, bytes));
    CHECK(ctx, gd_tensor_write(ctx, &g, g_data, bytes));
    CHECK(ctx, gd_context_seal_params(ctx));
    x.requires_grad = true;
    CHECK(ctx, gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(ctx, gd_relu(ctx, &x, &y));
    CHECK(ctx, gd_backward(ctx, &y, &g));
    CHECK(ctx, gd_tensor_grad(ctx, &x, &dx));
    CHECK(ctx, gd_end(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, bytes));
    if (write_file(argv[3], dx_data, bytes) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(g_data); free(dx_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_relu_bwd_runner.c"
    binary = tmp / "gd_relu_bwd_runner"
    source.write_text(RUNNER_SOURCE)
    cmd = ["cc", f"-I{root / 'include'}", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           str(source), str(root / "build" / "libgradients.a")]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary


def make_cases() -> list[Case]:
    smoke = [Case("edge_f16", 1, 8, True), Case("small_tail_f16", 4, 17)]
    classic = [Case("large_square_f16", 1024, 1024), Case("large_tail_f16", 2048, 513)]
    profile = os.environ.get("GD_RELU_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_RELU_BWD_PROFILE={profile!r}")
    return selected


def edge_input(rows: int, cols: int) -> torch.Tensor:
    bits = np.array([0x7E00, 0x8000, 0x0000, 0xBC00, 0x3C00, 0x7C00, 0xFC00, 0x0001], dtype=np.uint16)
    if rows * cols != bits.size:
        raise ValueError("edge case shape must match edge vector")
    return torch.from_numpy(bits.view(np.float16).reshape(rows, cols).copy())


def edge_grad(rows: int, cols: int) -> torch.Tensor:
    bits = np.array([0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x3800], dtype=np.uint16)
    if rows * cols != bits.size:
        raise ValueError("edge case shape must match edge vector")
    return torch.from_numpy(bits.view(np.float16).reshape(rows, cols).copy())


def case_tensors(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    if case.edge:
        return edge_input(case.rows, case.cols), edge_grad(case.rows, case.cols)
    gen = torch.Generator(device="cpu").manual_seed(2000 + case.rows * 5 + case.cols)
    x = (torch.randn(case.rows, case.cols, generator=gen, dtype=torch.float32) * 2.0 - 0.2).to(torch.float16)
    g = (torch.randn(case.rows, case.cols, generator=gen, dtype=torch.float32) * 0.5).to(torch.float16)
    return x, g


def f16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy().astype(np.float16, copy=False).view(np.uint16)


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    x, g = case_tensors(case)
    x_pt = x.clone().detach().requires_grad_()
    torch.relu(x_pt).backward(g)
    if x_pt.grad is None:
        raise RuntimeError("PyTorch autograd did not produce dx")
    ref = x_pt.grad
    x_path = tmp / f"{case.name}.x.bin"
    g_path = tmp / f"{case.name}.g.bin"
    dx_path = tmp / f"{case.name}.dx.bin"
    f16_bits(x).tofile(x_path)
    f16_bits(g).tofile(g_path)
    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run([str(runner), str(x_path), str(g_path), str(dx_path), str(case.rows), str(case.cols)],
                          cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 77:
        print(f"relu bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end=""); print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")
    actual_bits = np.fromfile(dx_path, dtype=np.uint16).reshape(case.rows, case.cols)
    ref_bits = f16_bits(ref).reshape(case.rows, case.cols)
    ok = bool(np.array_equal(actual_bits, ref_bits))
    mismatches = int(np.count_nonzero(actual_bits != ref_bits))
    print(f"relu bwd actual {'ok' if ok else 'FAIL'} case={case.name} dtype=f16 mismatches={mismatches}")
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-relu-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = sum(0 if run_case(runner, root, tmp, case) else 1 for case in make_cases())
        if failures:
            raise SystemExit(f"relu bwd check failed cases={failures}")


if __name__ == "__main__":
    main()
