#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Inspect gradients.c GPT-LM checkpoints.

Temporary diagnostic helper for examples/gpt_lm.  It parses the backend-neutral
.gdckpt container, prints metadata, tensor directory entries, and numeric stats
for checkpoint tensors without needing a gradients.c runtime context.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

MAGIC = b"GDCKPT1\0"
HEADER_STRUCT = struct.Struct("<8sIIQQQQQQ")
# path_len, flags, dtype, rank, 8 shape dims, data_offset, nbytes
DIR_FIXED_STRUCT = struct.Struct("<IIII8qQQ")
DTYPE_NAMES = {1: "f16", 2: "bf16", 3: "f32", 4: "i32", 5: "u8"}
DTYPE_NP = {1: np.dtype("<f2"), 3: np.dtype("<f4"), 4: np.dtype("<i4"), 5: np.dtype("u1")}
FLAG_NAMES = {1: "param", 2: "buffer"}


@dataclass(frozen=True)
class Header:
    version: int
    header_size: int
    tensor_count: int
    dir_offset: int
    dir_size: int
    metadata_offset: int
    metadata_len: int
    data_offset: int


@dataclass(frozen=True)
class Entry:
    path: str
    flags: int
    dtype: int
    rank: int
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int

    @property
    def dtype_name(self) -> str:
        return DTYPE_NAMES.get(self.dtype, f"dtype{self.dtype}")

    @property
    def flag_name(self) -> str:
        return FLAG_NAMES.get(self.flags, f"flags{self.flags}")

    @property
    def numel(self) -> int:
        out = 1
        for d in self.shape:
            out *= int(d)
        return out


def parse_header(f: BinaryIO) -> Header:
    raw = f.read(HEADER_STRUCT.size)
    if len(raw) != HEADER_STRUCT.size:
        raise ValueError("short checkpoint header")
    magic, version, header_size, tensor_count, dir_offset, dir_size, metadata_offset, metadata_len, data_offset = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"bad checkpoint magic: {magic!r}")
    return Header(version, header_size, tensor_count, dir_offset, dir_size, metadata_offset, metadata_len, data_offset)


def parse_entries(f: BinaryIO, h: Header) -> list[Entry]:
    entries: list[Entry] = []
    f.seek(h.dir_offset)
    consumed = 0
    for _ in range(h.tensor_count):
        raw = f.read(DIR_FIXED_STRUCT.size)
        if len(raw) != DIR_FIXED_STRUCT.size:
            raise ValueError("short checkpoint directory entry")
        path_len, flags, dtype, rank, *rest = DIR_FIXED_STRUCT.unpack(raw)
        dims = tuple(int(x) for x in rest[:8])
        data_offset = int(rest[8])
        nbytes = int(rest[9])
        path_raw = f.read(path_len)
        if len(path_raw) != path_len:
            raise ValueError("short checkpoint path")
        path = path_raw.decode("utf-8")
        entries.append(Entry(path, flags, dtype, rank, dims[:rank], data_offset, nbytes))
        consumed += DIR_FIXED_STRUCT.size + path_len
    if consumed != h.dir_size:
        raise ValueError(f"directory size mismatch: consumed={consumed} header={h.dir_size}")
    return entries


def read_metadata(f: BinaryIO, h: Header) -> str:
    if h.metadata_len == 0:
        return ""
    f.seek(h.metadata_offset)
    raw = f.read(h.metadata_len)
    if len(raw) != h.metadata_len:
        raise ValueError("short checkpoint metadata")
    return raw.decode("utf-8", errors="replace")


def read_entry_array(path: Path, e: Entry) -> np.ndarray:
    dtype = DTYPE_NP.get(e.dtype)
    if dtype is None:
        # bf16 and unknown dtypes are returned as raw u16/u8 for now.
        dtype = np.dtype("<u2") if e.dtype == 2 else np.dtype("u1")
    with path.open("rb") as f:
        f.seek(e.data_offset)
        raw = f.read(e.nbytes)
    if len(raw) != e.nbytes:
        raise ValueError(f"short tensor data for {e.path}")
    arr = np.frombuffer(raw, dtype=dtype)
    if e.dtype == 2:
        # bf16 -> float32
        bits = arr.astype(np.uint32) << np.uint32(16)
        arr = bits.view(np.float32)
    return arr.reshape(e.shape)


def stats_for_array(arr: np.ndarray) -> dict[str, float | int]:
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        return {"numel": 0}
    finite = np.isfinite(flat)
    out: dict[str, float | int] = {
        "numel": int(flat.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(flat).sum()),
        "inf": int(np.isinf(flat).sum()),
        "zero": int((flat == 0).sum()),
    }
    if finite.any():
        good = flat[finite]
        out.update(
            mean=float(good.mean()),
            std=float(good.std()),
            min=float(good.min()),
            p01=float(np.percentile(good, 1)),
            p50=float(np.percentile(good, 50)),
            p99=float(np.percentile(good, 99)),
            max=float(good.max()),
            rms=float(math.sqrt(float(np.mean(np.square(good, dtype=np.float64))))),
        )
    return out


def summarize_matrix(arr: np.ndarray) -> dict[str, float]:
    if arr.ndim != 2:
        return {}
    x = arr.astype(np.float32)
    row_norms = np.linalg.norm(x, axis=1)
    col_norms = np.linalg.norm(x, axis=0)
    return {
        "row_norm_min": float(row_norms.min()),
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_max": float(row_norms.max()),
        "col_norm_min": float(col_norms.min()),
        "col_norm_mean": float(col_norms.mean()),
        "col_norm_max": float(col_norms.max()),
    }


def parse_metadata_kv(metadata: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in metadata.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def inspect(path: Path, *, read_stats: bool, json_out: bool) -> None:
    with path.open("rb") as f:
        h = parse_header(f)
        entries = parse_entries(f, h)
        metadata = read_metadata(f, h)
    meta_kv = parse_metadata_kv(metadata)
    total_bytes = sum(e.nbytes for e in entries)
    total_numel = sum(e.numel for e in entries)
    payload_end = max((e.data_offset + e.nbytes for e in entries), default=h.data_offset)
    result: dict[str, object] = {
        "path": str(path),
        "header": h.__dict__,
        "metadata": meta_kv,
        "metadata_text": metadata,
        "tensor_count": len(entries),
        "total_numel": total_numel,
        "total_tensor_bytes": total_bytes,
        "payload_end": payload_end,
        "file_size": path.stat().st_size,
        "entries": [],
    }
    for e in entries:
        item: dict[str, object] = {
            "path": e.path,
            "kind": e.flag_name,
            "dtype": e.dtype_name,
            "shape": list(e.shape),
            "numel": e.numel,
            "nbytes": e.nbytes,
            "data_offset": e.data_offset,
        }
        if read_stats and e.dtype in (1, 2, 3, 4, 5):
            arr = read_entry_array(path, e)
            item["stats"] = stats_for_array(arr)
            if e.dtype in (1, 2, 3):
                mat_stats = summarize_matrix(arr)
                if mat_stats:
                    item["matrix"] = mat_stats
        result["entries"].append(item)
    if json_out:
        print(json.dumps(result, indent=2, sort_keys=False))
        return

    print(f"checkpoint: {path}")
    print(
        f"header: version={h.version} tensors={h.tensor_count} dir={h.dir_offset}+{h.dir_size} "
        f"metadata={h.metadata_offset}+{h.metadata_len} data_offset={h.data_offset}"
    )
    if metadata:
        print("metadata:")
        for line in metadata.splitlines():
            print(f"  {line}")
    print(
        f"tensors: count={len(entries)} total_numel={total_numel:,} "
        f"tensor_bytes={total_bytes/1024/1024:.2f} MiB file_size={path.stat().st_size/1024/1024:.2f} MiB"
    )
    for item in result["entries"]:  # type: ignore[index]
        shape = "x".join(str(x) for x in item["shape"])  # type: ignore[index]
        line = (
            f"  {item['path']:<64} {item['kind']:<6} {item['dtype']:<3} "
            f"[{shape:<12}] numel={item['numel']:<9} bytes={item['nbytes']}"
        )
        stats = item.get("stats") if isinstance(item, dict) else None
        if isinstance(stats, dict) and "mean" in stats:
            line += (
                f" mean={stats['mean']:.4g} std={stats['std']:.4g} "
                f"rms={stats['rms']:.4g} min={stats['min']:.4g} max={stats['max']:.4g}"
            )
            if stats.get("nan") or stats.get("inf"):
                line += f" NONFINITE nan={stats.get('nan')} inf={stats.get('inf')}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?", default="examples/gpt_lm/checkpoints/gpt_lm_latest.gdckpt")
    ap.add_argument("--no-stats", action="store_true", help="only parse metadata/directory")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    inspect(Path(args.checkpoint), read_stats=not args.no_stats, json_out=args.json)


if __name__ == "__main__":
    main()
