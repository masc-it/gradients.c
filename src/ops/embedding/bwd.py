# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

"""Live gradients.c embedding backward correctness check against PyTorch-style index_add."""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.op_oracle import build_library, compile_runner, gradients_env

import numpy as np
import torch


@dataclass(frozen=True)
class Case:
    name: str
    dtype: str
    vocab: int
    dim: int
    ids_shape: tuple[int, ...]


RUNNER_SOURCE = r'''
#include <gradients/gradients.h>

#include "tools/oracle_runner_common.c"

int main(int argc, char **argv)
{
    gd_dtype dtype;
    uint32_t vocab;
    uint32_t dim;
    uint32_t rank;
    int64_t ids_shape[GD_MAX_DIMS] = {0};
    int64_t table_shape[2];
    size_t ids_count = 1U;
    size_t table_count;
    size_t out_count;
    float *table_host = NULL;
    int32_t *ids_host = NULL;
    float *grad_host = NULL;
    float *dw_host = NULL;
    gd_context *ctx = NULL;
    gd_memory_config cfg;
    gd_tensor table, ids, grad_out, grad_table;
    uint32_t d;
    int arg;
    int rc = 1;

    if (argc < 10 || parse_dtype(argv[2], &dtype) != 0 || parse_u32(argv[3], &vocab) != 0 ||
        parse_u32(argv[4], &dim) != 0 || parse_u32(argv[5], &rank) != 0 || rank == 0U ||
        rank + 1U > GD_MAX_DIMS || argc != (int)(9U + rank)) {
        fprintf(stderr, "usage: %s OUT.bin f16|f32 VOCAB DIM IDS_RANK IDS_DIMS... TABLE.f32 IDS.i32 GRAD.f32\n", argv[0]);
        return 2;
    }
    arg = 6;
    for (d = 0U; d < rank; ++d) {
        if (parse_i64(argv[arg++], &ids_shape[d]) != 0 || ids_shape[d] <= 0 ||
            (uint64_t)ids_shape[d] > (uint64_t)(SIZE_MAX / ids_count)) { return 2; }
        ids_count *= (size_t)ids_shape[d];
    }
    if (dim == 0U || vocab == 0U || ids_count > SIZE_MAX / dim || (size_t)vocab > SIZE_MAX / dim) { return 2; }
    table_count = (size_t)vocab * dim;
    out_count = ids_count * dim;
    table_shape[0] = (int64_t)vocab;
    table_shape[1] = (int64_t)dim;
    table_host = (float *)malloc(sizeof(float) * table_count);
    ids_host = (int32_t *)malloc(sizeof(int32_t) * ids_count);
    grad_host = (float *)malloc(sizeof(float) * out_count);
    dw_host = (float *)malloc(sizeof(float) * table_count);
    if (table_host == NULL || ids_host == NULL || grad_host == NULL || dw_host == NULL ||
        read_file(argv[arg], table_host, sizeof(float) * table_count) != 0 ||
        read_file(argv[arg + 1], ids_host, sizeof(int32_t) * ids_count) != 0 ||
        read_file(argv[arg + 2], grad_host, sizeof(float) * out_count) != 0) { goto fail; }

    oracle_memory_config_slots(&cfg,
                               align_up((sizeof(float) * (table_count + out_count) + sizeof(int32_t) * ids_count) * 4U + 1024U * 1024U, 4096U),
                               align_up((sizeof(float) * table_count * 6U + sizeof(float) * out_count * 4U) + 1024U * 1024U, 4096U),
                               3U,
                               2U);
    if (check_status(NULL, gd_context_create(&cfg, &ctx), "gd_context_create") != 0) { goto fail; }
    CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(2U, table_shape), table_host, table_count, false, &table));
    CHECK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(rank, ids_shape), 256U, &ids));
    CHECK(ctx, gd_tensor_write(ctx, &ids, ids_host, sizeof(int32_t) * ids_count));
    {
        int64_t grad_shape[GD_MAX_DIMS] = {0};
        for (d = 0U; d < rank; ++d) { grad_shape[d] = ids_shape[d]; }
        grad_shape[rank] = (int64_t)dim;
        CHECK(ctx, gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, dtype, gd_shape_make(rank + 1U, grad_shape), grad_host, out_count, false, &grad_out));
    }
    CHECK(ctx, gd_context_seal_params(ctx));
    CHECK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(ctx, gd_embedding_backward(ctx, &table, &ids, &grad_out, &grad_table));
    CHECK(ctx, gd_end_step(ctx));
    CHECK(ctx, gd_synchronize(ctx));
    CHECK(ctx, gd_tensor_read_f32(ctx, &grad_table, dw_host, table_count));
    if (write_file(argv[1], dw_host, sizeof(float) * table_count) != 0) { goto fail; }
    rc = 0;
    goto done;
fail:
    rc = 1;
done:
    gd_context_destroy(ctx);
    free(table_host); free(ids_host); free(grad_host); free(dw_host);
    return rc;
}
'''


def cases() -> list[Case]:
    all_cases = [
        Case("tiny_f32", "f32", 7, 5, (2, 3)),
        Case("tokens_f16", "f16", 257, 64, (4, 17)),
        Case("lm_f16", "f16", 4096, 768, (8, 128)),
        Case("wide_f32", "f32", 1024, 256, (16, 32)),
    ]
    profile = os.environ.get("GD_EMBEDDING_BWD_PROFILE", "smoke")
    if profile == "smoke":
        return all_cases[:2]
    if profile == "all":
        return all_cases
    selected = [case for case in all_cases if case.name == profile]
    if not selected:
        raise SystemExit(f"unknown GD_EMBEDDING_BWD_PROFILE={profile!r}")
    return selected


def torch_dtype(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "f32": torch.float32}[dtype]


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = (sum((i + 1) * ord(ch) for i, ch in enumerate(case.name)) ^ 0x454D4242) & 0xFFFF_FFFF
    gen = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.randn((case.vocab, case.dim), generator=gen, dtype=torch.float32) * 0.25
    ids = torch.randint(0, case.vocab, case.ids_shape, generator=gen, dtype=torch.int32)
    if ids.numel() >= 8:
        flat = ids.reshape(-1)
        flat[1] = flat[0]
        flat[5] = flat[0]
    grad = torch.randn((*case.ids_shape, case.dim), generator=gen, dtype=torch.float32) * 0.1
    return table, ids, grad


def reference(case: Case, ids: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    dtype = torch_dtype(case.dtype)
    grad_q = grad.to(dtype)
    want = torch.zeros((case.vocab, case.dim), dtype=torch.float32)
    want.index_add_(0, ids.reshape(-1).long(), grad_q.reshape(-1, case.dim).float())
    return want.to(dtype).float() if dtype == torch.float16 else want


def run_case(binary: Path, root: Path, tmp: Path, case: Case) -> None:
    table, ids, grad = make_inputs(case)
    want = reference(case, ids, grad)
    table_path = tmp / f"{case.name}_table.f32"
    ids_path = tmp / f"{case.name}_ids.i32"
    grad_path = tmp / f"{case.name}_grad.f32"
    out_path = tmp / f"{case.name}_dw.f32"
    table_path.write_bytes(table.contiguous().numpy().astype(np.float32).tobytes())
    ids_path.write_bytes(ids.contiguous().numpy().astype(np.int32).tobytes())
    grad_path.write_bytes(grad.contiguous().numpy().astype(np.float32).tobytes())
    cmd = [str(binary), str(out_path), case.dtype, str(case.vocab), str(case.dim), str(len(case.ids_shape)),
           *map(str, case.ids_shape), str(table_path), str(ids_path), str(grad_path)]
    env = gradients_env(root)
    subprocess.run(cmd, check=True, env=env)
    got = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(case.vocab, case.dim)
    atol = 2.0e-3 if case.dtype == "f16" else 1.0e-5
    rtol = 2.0e-3 if case.dtype == "f16" else 1.0e-5
    np.testing.assert_allclose(got, want.numpy(), atol=atol, rtol=rtol)
    print(f"[embedding/bwd] {case.name}: ok")


def main() -> None:
    root = _REPO_ROOT
    build_library(root)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        binary = compile_runner(root, tmp, "gd_embedding_bwd_runner", RUNNER_SOURCE)
        for case in cases():
            run_case(binary, root, tmp, case)


if __name__ == "__main__":
    main()
