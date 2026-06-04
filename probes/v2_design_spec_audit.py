#!/usr/bin/env python3
"""Static v2 design-spec coverage audit for gradients.c.

This probe scans source/docs for implementation coverage markers. It is meant to
complement runtime probes: current step-by-step work may have many MISS rows.
Use --strict when CI should fail on missing future-scope items.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def tree_text(rels: tuple[str, ...], suffixes: set[str]) -> str:
    chunks: list[str] = []
    for rel in rels:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in suffixes:
                chunks.append(f"\n/* {path.relative_to(ROOT)} */\n")
                chunks.append(read_text(path))
    return "\n".join(chunks)


IMPL = tree_text(("include", "src"), {".h", ".c", ".m", ".metal"})
TESTS = tree_text(("tests",), {".h", ".c", ".m", ".metal"})
DESIGN = read_text(ROOT / "docs" / "design_spec.md")
README = read_text(ROOT / "README.md")


class Row:
    def __init__(self, area: str, item: str, status: str, evidence: str) -> None:
        self.area = area
        self.item = item
        self.status = status
        self.evidence = evidence


def has_all(needles: Iterable[str], haystack: str = IMPL) -> bool:
    return all(needle in haystack for needle in needles)


def has_any(needles: Iterable[str], haystack: str = IMPL) -> bool:
    return any(needle in haystack for needle in needles)


def op_capsule_exists(name: str) -> bool:
    return (ROOT / "src" / "ops" / name).is_dir()


def add_symbol(rows: list[Row], area: str, item: str, symbols: list[str]) -> None:
    ok = has_all(symbols)
    rows.append(Row(area, item, "PASS" if ok else "MISS", ", ".join(symbols)))


def add_op(rows: list[Row], name: str) -> None:
    ok = op_capsule_exists(name)
    rows.append(Row("ops", f"src/ops/{name}", "PASS" if ok else "MISS", "op capsule directory"))


def add_check(rows: list[Row], area: str, item: str, ok: bool, evidence: str) -> None:
    rows.append(Row(area, item, "PASS" if ok else "MISS", evidence))


def collect_rows() -> list[Row]:
    rows: list[Row] = []

    add_symbol(rows, "memory", "context + scoped eager lifecycle", [
        "gd_context_create", "gd_begin", "gd_end", "gd_memory_stats_query",
    ])
    add_symbol(rows, "memory", "four logical arenas", [
        "GD_ARENA_PARAMS", "GD_ARENA_STATE", "GD_ARENA_SCRATCH", "GD_ARENA_DATA",
    ])
    add_symbol(rows, "memory", "ring debug generations/fences", [
        "gd_debug_current_ring_slot", "gd_debug_ring_slot_generation", "gd_debug_ring_slot_fence",
    ])
    add_symbol(rows, "memory", "state object fence reset/reuse", [
        "gd_state_object_create", "gd_state_object_acquire_span", "gd_state_object_reset",
    ])
    add_symbol(rows, "tensor", "concrete tensor descriptors + views", [
        "gd_tensor_empty", "gd_tensor_slice", "gd_tensor_contiguous", "gd_tensor_validate",
    ])
    add_symbol(rows, "tensor", "GPU init helpers", [
        "gd_tensor_zeros", "gd_tensor_ones", "gd_tensor_rand_uniform", "gd_tensor_zero_",
    ])
    add_symbol(rows, "transfer", "explicit span/tensor transfer path", [
        "gd_span_upload", "gd_span_download", "gd_upload", "gd_download", "gd_synchronize",
    ])
    add_symbol(rows, "backend", "buffer/fence backend portability layer", [
        "gd_backend_buffer", "gd_backend_fence", "gd_backend_record_fence", "gd_backend_upload",
    ])
    add_symbol(rows, "backend", "stream/kernel/matmul abstractions", [
        "gd_backend_stream", "gd_kernel", "gd_matmul",
    ])
    add_check(rows, "backend", "Metal shared tensor arenas", "MTLResourceStorageModeShared" in IMPL,
              "MTLResourceStorageModeShared")
    add_check(rows, "backend", "no Metal private tensor arena marker", "MTLResourceStorageModePrivate" not in IMPL,
              "MTLResourceStorageModePrivate absent")
    add_check(rows, "backend", "no CPU backend directory", not (ROOT / "src" / "backends" / "cpu").exists(),
              "src/backends/cpu absent")
    add_check(rows, "docs", "README still references v1 CPU/graph language",
              not has_any(["CPU reference execution", "graph capture"], README),
              "README should match v2 GPU-only/no public graph direction")

    for op in [
        "add", "matmul", "rms_norm", "embedding", "sdpa_varlen",
        "sdpa_cached_prefill", "sdpa_cached_decode", "kv_cache_write_at",
        "lm_cross_entropy_suffix", "loss_sum", "argmax", "topk", "topk_sample",
        "concat", "slice", "gather_positions", "pool_mean_masked", "contrastive_loss",
    ]:
        add_op(rows, op)

    add_symbol(rows, "autograd", "tape/backward public markers", [
        "gd_amp_backward", "backward", "tape",
    ])
    add_symbol(rows, "optim", "optimizer + AMP scaler", [
        "gd_optimizer_zero_grad", "gd_optimizer_step_amp", "scaler",
    ])
    add_symbol(rows, "checkpoint", "training checkpoint save/load", [
        "gd_checkpoint_save", "gd_checkpoint_load", "gd_checkpoint_state",
    ])
    add_symbol(rows, "modules", "Module/ModuleList/ModuleDict", [
        "gd_module", "gd_module_list", "gd_module_dict",
    ])
    add_symbol(rows, "models", "early-fusion VLM/GPT stress markers", [
        "prefix_len", "lm_cross_entropy_suffix", "ModuleDict",
    ])

    add_check(rows, "docs", "design spec present", "# gradients.c v2 design spec" in DESIGN,
              "docs/design_spec.md")
    return rows


def print_report(rows: list[Row]) -> None:
    counts = {"PASS": 0, "MISS": 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    print("# v2 design-spec static audit")
    print()
    print(f"root: `{ROOT}`")
    print(f"summary: PASS={counts.get('PASS', 0)} MISS={counts.get('MISS', 0)}")
    print()
    print("| status | area | item | evidence |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print(f"| {row.status} | {row.area} | {row.item} | `{row.evidence}` |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit nonzero when any MISS row exists")
    args = parser.parse_args()
    rows = collect_rows()
    print_report(rows)
    if args.strict and any(row.status == "MISS" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
