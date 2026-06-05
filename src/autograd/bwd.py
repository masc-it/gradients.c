# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c autograd correctness check against PyTorch autograd.

The checked graph intentionally has fanout and multiple losses:

    y  = matmul(x, w)
    z1 = linear(y, v, b)
    z2 = matmul(y, u)
    backward_many([z1, z2], [g1, g2])

This exercises tape recording, reverse dispatch through per-op autograd capsules,
and backend-side gradient accumulation into y before the matmul backward rule.

Run from repo root:

    uv run src/autograd/bwd.py

Knobs:

    GD_AUTOGRAD_BWD_PROFILE=smoke|classic|all|<case>   # default: smoke
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
    p: int
    q: int


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
    if (fread(dst, 1U, nbytes, f) != nbytes) { fprintf(stderr, "short read %s\n", path); fclose(f); return 1; }
    return fclose(f) != 0 ? 1 : 0;
}

static int write_file(const char *path, const void *src, size_t nbytes)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    if (fwrite(src, 1U, nbytes, f) != nbytes) { fprintf(stderr, "short write %s\n", path); fclose(f); return 1; }
    return fclose(f) != 0 ? 1 : 0;
}

static int parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v;
    if (s == NULL || out == NULL || s[0] == '\0') { return 1; }
    v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v == 0UL || v > (unsigned long)UINT32_MAX) { return 1; }
    *out = (uint32_t)v;
    return 0;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX / b) { return 1; }
    *out = a * b;
    return 0;
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

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
    uint32_t m, k, n, p, q;
    size_t x_e, w_e, v_e, b_e, u_e, g1_e, g2_e;
    size_t x_b, w_b, v_b, b_b, u_b, g1_b, g2_b;
    uint16_t *x_data = NULL, *w_data = NULL, *v_data = NULL, *b_data = NULL, *u_data = NULL;
    uint16_t *g1_data = NULL, *g2_data = NULL;
    uint16_t *dx_data = NULL, *dw_data = NULL, *dv_data = NULL, *db_data = NULL, *du_data = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor x, w, v, b, u, g1, g2, y, z1, z2, dx, dw, dv, db, du;
    gd_tensor dummy;
    const gd_tensor *outputs[2];
    const gd_tensor *grad_outputs[2];
    int64_t x_shape[2], w_shape[2], v_shape[2], b_shape[1], u_shape[2], g1_shape[2], g2_shape[2];
    gd_status st;
    int rc = 1;

    if (argc != 18) {
        fprintf(stderr, "usage: %s X W V B U G1 G2 DX DW DV DB DU M K N P Q\n", argv[0]);
        return 2;
    }
    if (parse_u32(argv[13], &m) != 0 || parse_u32(argv[14], &k) != 0 ||
        parse_u32(argv[15], &n) != 0 || parse_u32(argv[16], &p) != 0 ||
        parse_u32(argv[17], &q) != 0) {
        return 2;
    }
    if (mul_size((size_t)m, (size_t)k, &x_e) != 0 || mul_size((size_t)k, (size_t)n, &w_e) != 0 ||
        mul_size((size_t)n, (size_t)p, &v_e) != 0 || mul_size((size_t)p, 1U, &b_e) != 0 ||
        mul_size((size_t)n, (size_t)q, &u_e) != 0 || mul_size((size_t)m, (size_t)p, &g1_e) != 0 ||
        mul_size((size_t)m, (size_t)q, &g2_e) != 0 || mul_size(x_e, 2U, &x_b) != 0 ||
        mul_size(w_e, 2U, &w_b) != 0 || mul_size(v_e, 2U, &v_b) != 0 ||
        mul_size(b_e, 2U, &b_b) != 0 || mul_size(u_e, 2U, &u_b) != 0 ||
        mul_size(g1_e, 2U, &g1_b) != 0 || mul_size(g2_e, 2U, &g2_b) != 0) {
        return 2;
    }

    x_data = (uint16_t *)malloc(x_b); w_data = (uint16_t *)malloc(w_b); v_data = (uint16_t *)malloc(v_b);
    b_data = (uint16_t *)malloc(b_b); u_data = (uint16_t *)malloc(u_b); g1_data = (uint16_t *)malloc(g1_b);
    g2_data = (uint16_t *)malloc(g2_b); dx_data = (uint16_t *)calloc(x_e, sizeof(*dx_data));
    dw_data = (uint16_t *)calloc(w_e, sizeof(*dw_data)); dv_data = (uint16_t *)calloc(v_e, sizeof(*dv_data));
    db_data = (uint16_t *)calloc(b_e, sizeof(*db_data)); du_data = (uint16_t *)calloc(u_e, sizeof(*du_data));
    if (x_data == NULL || w_data == NULL || v_data == NULL || b_data == NULL || u_data == NULL ||
        g1_data == NULL || g2_data == NULL || dx_data == NULL || dw_data == NULL || dv_data == NULL ||
        db_data == NULL || du_data == NULL) { goto fail; }
    if (read_file(argv[1], x_data, x_b) != 0 || read_file(argv[2], w_data, w_b) != 0 ||
        read_file(argv[3], v_data, v_b) != 0 || read_file(argv[4], b_data, b_b) != 0 ||
        read_file(argv[5], u_data, u_b) != 0 || read_file(argv[6], g1_data, g1_b) != 0 ||
        read_file(argv[7], g2_data, g2_b) != 0) { goto fail; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(x_b + w_b + v_b + b_b + u_b + g1_b + g2_b + 8U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = 128U * 1024U * 1024U;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) { printf("SKIP unsupported backend\n"); rc = 77; goto done; }
    if (check_status(ctx, st, "gd_context_create") != 0 || ctx == NULL) { goto fail; }

    x_shape[0] = (int64_t)m; x_shape[1] = (int64_t)k;
    w_shape[0] = (int64_t)k; w_shape[1] = (int64_t)n;
    v_shape[0] = (int64_t)n; v_shape[1] = (int64_t)p;
    b_shape[0] = (int64_t)p;
    u_shape[0] = (int64_t)n; u_shape[1] = (int64_t)q;
    g1_shape[0] = (int64_t)m; g1_shape[1] = (int64_t)p;
    g2_shape[0] = (int64_t)m; g2_shape[1] = (int64_t)q;

    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, v_shape), 256U, &v));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, b_shape), 256U, &b));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, u_shape), 256U, &u));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, g1_shape), 256U, &g1));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, g2_shape), 256U, &g2));
    CHECK(ctx, gd_tensor_write(ctx, &x, x_data, x_b));
    CHECK(ctx, gd_tensor_write(ctx, &w, w_data, w_b));
    CHECK(ctx, gd_tensor_write(ctx, &v, v_data, v_b));
    CHECK(ctx, gd_tensor_write(ctx, &b, b_data, b_b));
    CHECK(ctx, gd_tensor_write(ctx, &u, u_data, u_b));
    CHECK(ctx, gd_tensor_write(ctx, &g1, g1_data, g1_b));
    CHECK(ctx, gd_tensor_write(ctx, &g2, g2_data, g2_b));
    CHECK(ctx, gd_context_seal_params(ctx));

    x.requires_grad = true; w.requires_grad = true; v.requires_grad = true; b.requires_grad = true; u.requires_grad = true;
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_matmul(ctx, &x, &w, &y));
    CHECK(ctx, gd_linear(ctx, &y, &v, &b, &z1));
    CHECK(ctx, gd_matmul(ctx, &y, &u, &z2));
    outputs[0] = &z1; outputs[1] = &z2;
    grad_outputs[0] = &g1; grad_outputs[1] = &g2;
    CHECK(ctx, gd_backward_many(ctx, 2U, outputs, grad_outputs));
    CHECK(ctx, gd_tensor_grad(ctx, &x, &dx));
    CHECK(ctx, gd_tensor_grad(ctx, &w, &dw));
    CHECK(ctx, gd_tensor_grad(ctx, &v, &dv));
    CHECK(ctx, gd_tensor_grad(ctx, &b, &db));
    CHECK(ctx, gd_tensor_grad(ctx, &u, &du));
    CHECK(ctx, gd_tensor_grad(ctx, &y, &dummy));
    (void)dummy;
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read(ctx, &dx, dx_data, x_b));
    CHECK(ctx, gd_tensor_read(ctx, &dw, dw_data, w_b));
    CHECK(ctx, gd_tensor_read(ctx, &dv, dv_data, v_b));
    CHECK(ctx, gd_tensor_read(ctx, &db, db_data, b_b));
    CHECK(ctx, gd_tensor_read(ctx, &du, du_data, u_b));
    if (write_file(argv[8], dx_data, x_b) != 0 || write_file(argv[9], dw_data, w_b) != 0 ||
        write_file(argv[10], dv_data, v_b) != 0 || write_file(argv[11], db_data, b_b) != 0 ||
        write_file(argv[12], du_data, u_b) != 0) { goto fail; }
    rc = 0;
    goto done;

fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(x_data); free(w_data); free(v_data); free(b_data); free(u_data); free(g1_data); free(g2_data);
    free(dx_data); free(dw_data); free(dv_data); free(db_data); free(du_data);
    return rc;
}
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_library(root: Path) -> None:
    subprocess.run(["make", "build"], cwd=root, check=True)


def compile_runner(root: Path, tmp: Path) -> Path:
    source = tmp / "gd_autograd_bwd_runner.c"
    binary = tmp / "gd_autograd_bwd_runner"
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
    smoke = [Case("fanout_small", 4, 7, 6, 5, 3)]
    classic = [
        Case("aligned_32", 32, 32, 32, 32, 32),
        Case("aligned_64", 64, 64, 64, 64, 64),
    ]
    profile = os.environ.get("GD_AUTOGRAD_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return smoke
    if profile == "classic":
        return classic
    if profile == "all":
        return smoke + classic
    selected = [case for case in smoke + classic if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_AUTOGRAD_BWD_PROFILE={profile!r}")
    return selected


def write_f16(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().view(np.uint16).tofile(path)


def read_f16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16)
    want = int(np.prod(shape))
    if bits.size != want:
        raise RuntimeError(f"bad output size for {path}: got {bits.size}, want {want}")
    return torch.from_numpy(bits.view(np.float16).reshape(shape).copy()).to(torch.float32)


def gemm_store_f16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(torch.float32) @ b.to(torch.float32)).to(torch.float16).to(torch.float32)


def reduce_rows_store_f16(a: torch.Tensor) -> torch.Tensor:
    return a.to(torch.float32).sum(dim=0).to(torch.float16).to(torch.float32)


def add_store_f16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(torch.float16) + b.to(torch.float16)).to(torch.float16).to(torch.float32)


def check_close(name: str, actual: torch.Tensor, ref: torch.Tensor) -> tuple[bool, str]:
    diff = (actual - ref).abs()
    rel = diff / torch.maximum(ref.abs(), torch.tensor(1.0e-6, dtype=torch.float32))
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    ok = bool(torch.all(diff <= (1.2e-2 + 8.0e-2 * ref.abs())).item())
    return ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} mean_abs={mean_abs:.3e}"


def run_case(runner: Path, root: Path, tmp: Path, case: Case) -> bool:
    seed = 9000 + case.m * 3 + case.k * 5 + case.n * 7 + case.p * 11 + case.q * 13
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(case.m, case.k, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    w = (torch.randn(case.k, case.n, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    v = (torch.randn(case.n, case.p, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    b = (torch.randn(case.p, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    u = (torch.randn(case.n, case.q, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    g1 = (torch.randn(case.m, case.p, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)
    g2 = (torch.randn(case.m, case.q, generator=gen, dtype=torch.float32) * 0.04).to(torch.float16)

    x_pt = x.clone().detach().requires_grad_()
    w_pt = w.clone().detach().requires_grad_()
    v_pt = v.clone().detach().requires_grad_()
    b_pt = b.clone().detach().requires_grad_()
    u_pt = u.clone().detach().requires_grad_()
    y_pt = x_pt @ w_pt
    z1_pt = y_pt @ v_pt + b_pt
    z2_pt = y_pt @ u_pt
    loss_pt = (z1_pt * g1).sum() + (z2_pt * g2).sum()
    loss_pt.backward()
    if x_pt.grad is None or w_pt.grad is None or v_pt.grad is None or b_pt.grad is None or u_pt.grad is None:
        raise RuntimeError("PyTorch autograd did not produce all gradients")
    dx_ref = x_pt.grad.to(torch.float32)
    dw_ref = w_pt.grad.to(torch.float32)
    dv_ref = v_pt.grad.to(torch.float32)
    db_ref = b_pt.grad.to(torch.float32)
    du_ref = u_pt.grad.to(torch.float32)

    paths = {
        "x": tmp / f"{case.name}.x.f16",
        "w": tmp / f"{case.name}.w.f16",
        "v": tmp / f"{case.name}.v.f16",
        "b": tmp / f"{case.name}.b.f16",
        "u": tmp / f"{case.name}.u.f16",
        "g1": tmp / f"{case.name}.g1.f16",
        "g2": tmp / f"{case.name}.g2.f16",
        "dx": tmp / f"{case.name}.dx.f16",
        "dw": tmp / f"{case.name}.dw.f16",
        "dv": tmp / f"{case.name}.dv.f16",
        "db": tmp / f"{case.name}.db.f16",
        "du": tmp / f"{case.name}.du.f16",
    }
    for key, tensor in (("x", x), ("w", w), ("v", v), ("b", b), ("u", u), ("g1", g1), ("g2", g2)):
        write_f16(paths[key], tensor)

    env = os.environ.copy()
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    proc = subprocess.run(
        [
            str(runner),
            str(paths["x"]), str(paths["w"]), str(paths["v"]), str(paths["b"]), str(paths["u"]),
            str(paths["g1"]), str(paths["g2"]), str(paths["dx"]), str(paths["dw"]),
            str(paths["dv"]), str(paths["db"]), str(paths["du"]),
            str(case.m), str(case.k), str(case.n), str(case.p), str(case.q),
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 77:
        print(f"autograd bwd skipped case={case.name}: {proc.stdout.strip()}")
        return True
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        raise RuntimeError(f"runner failed case={case.name} rc={proc.returncode}")

    actuals = {
        "dx": read_f16(paths["dx"], (case.m, case.k)),
        "dw": read_f16(paths["dw"], (case.k, case.n)),
        "dv": read_f16(paths["dv"], (case.n, case.p)),
        "db": read_f16(paths["db"], (case.p,)),
        "du": read_f16(paths["du"], (case.n, case.q)),
    }
    refs = {"dx": dx_ref, "dw": dw_ref, "dv": dv_ref, "db": db_ref, "du": du_ref}
    ok = True
    msgs: list[str] = []
    for name in ("dx", "dw", "dv", "db", "du"):
        name_ok, msg = check_close(name, actuals[name], refs[name])
        ok = ok and name_ok
        msgs.append(msg)
    print(
        f"autograd bwd actual {'ok' if ok else 'FAIL'} case={case.name} "
        f"shape=({case.m},{case.k})x({case.k},{case.n})->p{case.p}/q{case.q} "
        + " ".join(msgs)
    )
    return ok


def main() -> None:
    root = repo_root()
    build_library(root)
    with tempfile.TemporaryDirectory(prefix="gd-autograd-bwd-") as tmp_str:
        tmp = Path(tmp_str)
        runner = compile_runner(root, tmp)
        failures = 0
        for case in make_cases():
            if not run_case(runner, root, tmp, case):
                failures += 1
        if failures:
            raise SystemExit(f"autograd bwd actual check failed cases={failures}")


if __name__ == "__main__":
    main()
