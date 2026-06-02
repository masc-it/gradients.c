#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyarrow",
#     "numpy",
#     "pillow",
#     "tqdm",
# ]
# ///
"""Preprocess ImageNet enriched parquet into gradients.c GPT-VLM shards.

Expected pipeline:
  1) extract_label_text.py -> labels.txt + label_text.txt
  2) gradients-train-bpe -> tokenizer.json
  3) tokenize_label_text.py -> text-tokenized.json
  4) this script -> FP16 patch shards + token-id records

Output record format (gd-imagenet-vlm-shard-v1):
  [128B shard header]
  repeated samples:
    int32 label_id
    uint32 token_len
    token ids, little-endian u16/u32 selected from tokenizer vocab size
    float16 patches [196, 768] in CHW-patch-major order

Index files contain fixed-width 32B entries with body offsets for random access.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import re
import struct
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, PngImagePlugin
from tqdm import tqdm

# Some ImageNet PNGs carry large metadata chunks.
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024

DEFAULT_IN_DIR = "/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched"
DEFAULT_OUT_DIR = (
    "/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-"
    "gradients-224-16patch"
)

TARGET_H = 224
TARGET_W = 224
CHANNELS = 3
PATCH_SIZE = 16
NUM_PATCHES = (TARGET_H // PATCH_SIZE) * (TARGET_W // PATCH_SIZE)
PATCH_DIM = CHANNELS * PATCH_SIZE * PATCH_SIZE
PATCH_DTYPE_F16 = 1
TOKEN_DTYPE_U16 = 1
TOKEN_DTYPE_U32 = 2
PATCH_BYTES_PER_SAMPLE = NUM_PATCHES * PATCH_DIM * np.dtype("<f2").itemsize
RESIZE_SHORT = round(TARGET_H * 256 / 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SHARD_MAGIC = b"GDVLMv1\0"
SHARD_VERSION = 1
SHARD_HEADER_SIZE = 128
# magic, version, header_size, H, W, C, patch_size, num_patches, patch_dim,
# patch_dtype, token_dtype, vocab_size, shard_idx, num_shards, payload_offset,
# n_samples, tokenizer_hash
SHARD_HEADER_STRUCT = struct.Struct("<8sIIIIIIIIIIIIIIQQ")
SHARD_NUM_SHARDS_OFFSET = 56

IDX_MAGIC = b"GDVLMIDX"
IDX_VERSION = 1
IDX_HEADER_STRUCT = struct.Struct("<8sIIQ")
# shard_idx, sample_idx, body_offset, record_nbytes, token_len, label_id, flags, raw_pos
IDX_ENTRY_STRUCT = struct.Struct("<IIQIIHHI")
IDX_ENTRY_BYTES = IDX_ENTRY_STRUCT.size

SHARD_NAME_RE = re.compile(
    r"^(?P<split>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.(?P<ext>gdvlm|idx)$"
)

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - older Pillow.
    RESAMPLE_BICUBIC = Image.BICUBIC


def _token_dtype_for_vocab(vocab_size: int) -> int:
    return TOKEN_DTYPE_U16 if vocab_size <= 0xFFFF else TOKEN_DTYPE_U32


def _token_dtype_name(token_dtype: int) -> str:
    if token_dtype == TOKEN_DTYPE_U16:
        return "u16"
    if token_dtype == TOKEN_DTYPE_U32:
        return "u32"
    raise ValueError(f"bad token dtype: {token_dtype}")


def _token_dtype_size(token_dtype: int) -> int:
    if token_dtype == TOKEN_DTYPE_U16:
        return 2
    if token_dtype == TOKEN_DTYPE_U32:
        return 4
    raise ValueError(f"bad token dtype: {token_dtype}")


def _pack_token_ids(token_ids: list[int], token_dtype: int) -> bytes:
    if token_dtype == TOKEN_DTYPE_U16:
        if any(t < 0 or t > 0xFFFF for t in token_ids):
            raise ValueError("token id out of u16 range")
        return struct.pack(f"<{len(token_ids)}H", *token_ids)
    if token_dtype == TOKEN_DTYPE_U32:
        if any(t < 0 or t > 0xFFFFFFFF for t in token_ids):
            raise ValueError("token id out of u32 range")
        return struct.pack(f"<{len(token_ids)}I", *token_ids)
    raise ValueError(f"bad token dtype: {token_dtype}")


def _decode_hash_u64(value: Any) -> int:
    if isinstance(value, int):
        return value & 0xFFFFFFFFFFFFFFFF
    if isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        return int(s, 16) & 0xFFFFFFFFFFFFFFFF
    raise ValueError(f"invalid tokenizer_hash: {value!r}")


def load_tokenized_text(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "gd-imagenet-vlm-text-tokenized-v1":
        raise ValueError(f"bad tokenized text format in {path}")
    vocab_size = int(data["vocab_size"])
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("text-tokenized.json missing non-empty labels list")
    by_id: dict[int, dict[str, Any]] = {}
    for entry in labels:
        label_id = int(entry["label_id"])
        ids = [int(x) for x in entry["ids"]]
        if label_id in by_id:
            raise ValueError(f"duplicate label_id in tokenized text: {label_id}")
        if not ids:
            raise ValueError(f"empty token ids for label_id={label_id}")
        if any(t < 0 or t >= vocab_size for t in ids):
            raise ValueError(f"token id outside vocab for label_id={label_id}")
        by_id[label_id] = {**entry, "ids": ids, "n_tokens": len(ids)}
    return {
        "raw": data,
        "by_id": by_id,
        "vocab_size": vocab_size,
        "tokenizer_hash_u64": _decode_hash_u64(data["tokenizer_hash"]),
        "tokenizer_hash": str(data["tokenizer_hash"]),
        "tokenizer_path": str(data.get("tokenizer_path", "")),
        "pad_id": int(data.get("pad_id", -1)),
    }


def _image_bytes_pylist(table: Any) -> list[Any]:
    col = table.column("image").combine_chunks()
    if hasattr(col, "field"):
        names = getattr(col.type, "names", [])
        if "bytes" in names:
            return col.field("bytes").to_pylist()
    return col.to_pylist()


def _coerce_image_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def process_sample(args: tuple[int, bytes, int, list[int], int]) -> tuple[Any, ...]:
    """Decode/patch one sample. Returns (raw_pos, blob, label_id, token_len)."""
    raw_pos, img_bytes, label_id, token_ids, token_dtype = args
    try:
        img_pil = Image.open(io.BytesIO(img_bytes))
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        w, h = img_pil.size
        short = min(w, h)
        if short <= 0:
            return (raw_pos, None, "invalid image size")
        scale = float(RESIZE_SHORT) / float(short)
        new_w = max(TARGET_W, int(w * scale))
        new_h = max(TARGET_H, int(h * scale))
        img_pil = img_pil.resize((new_w, new_h), RESAMPLE_BICUBIC)

        left = (new_w - TARGET_W) // 2
        top = (new_h - TARGET_H) // 2
        img_pil = img_pil.crop((left, top, left + TARGET_W, top + TARGET_H))

        img = np.asarray(img_pil, dtype=np.float32) * (1.0 / 255.0)  # HWC
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(img, (2, 0, 1))  # C,H,W
        patches = (
            chw.reshape(
                CHANNELS,
                TARGET_H // PATCH_SIZE,
                PATCH_SIZE,
                TARGET_W // PATCH_SIZE,
                PATCH_SIZE,
            )
            .transpose(1, 3, 0, 2, 4)
            .reshape(NUM_PATCHES, PATCH_DIM)
            .astype("<f2", copy=False)
        )
        patch_bytes = patches.tobytes(order="C")
        if len(patch_bytes) != PATCH_BYTES_PER_SAMPLE:
            return (raw_pos, None, "bad patch byte count")

        token_bytes = _pack_token_ids(token_ids, token_dtype)
        blob = (
            struct.pack("<iI", int(label_id), len(token_ids))
            + token_bytes
            + patch_bytes
        )
        return (raw_pos, blob, int(label_id), len(token_ids))
    except Exception as exc:  # Keep bad images from killing full preprocessing.
        return (raw_pos, None, repr(exc))


def _shard_path(out_dir: Path, split_tag: str, shard_idx: int, num_shards: int, ext: str) -> Path:
    return out_dir / f"{split_tag}-{shard_idx + 1:05d}-of-{num_shards:05d}.{ext}"


def _parse_shard_name(filename: str, split_tag: str, ext: str) -> tuple[int, int] | None:
    m = SHARD_NAME_RE.match(filename)
    if not m or m.group("split") != split_tag or m.group("ext") != ext:
        return None
    return int(m.group("idx")), int(m.group("total"))


def _idx_entry_count(path: Path) -> int:
    return path.stat().st_size // IDX_ENTRY_BYTES


def _meta_path(out_dir: Path, split_tag: str) -> Path:
    return out_dir / f"{split_tag}.meta.json"


def _read_meta(out_dir: Path, split_tag: str) -> dict[str, Any] | None:
    path = _meta_path(out_dir, split_tag)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_split_outputs(out_dir: Path, split_tag: str) -> None:
    if not out_dir.is_dir():
        return
    for path in out_dir.iterdir():
        parsed_gdvlm = _parse_shard_name(path.name, split_tag, "gdvlm")
        parsed_idx = _parse_shard_name(path.name, split_tag, "idx")
        if (
            path.name in (f"{split_tag}.idx", f"{split_tag}.idx.tmp", f"{split_tag}.meta.json", f"{split_tag}.meta.json.tmp")
            or parsed_gdvlm
            or parsed_idx
            or (path.name.startswith(f"{split_tag}-") and path.name.endswith((".gdvlm.tmp", ".idx.tmp")))
        ):
            path.unlink()


def _clean_stale_tmp_outputs(out_dir: Path, split_tag: str) -> None:
    if not out_dir.is_dir():
        return
    for path in out_dir.iterdir():
        if path.name in (f"{split_tag}.idx.tmp", f"{split_tag}.meta.json.tmp") or (
            path.name.startswith(f"{split_tag}-") and path.name.endswith((".gdvlm.tmp", ".idx.tmp"))
        ):
            path.unlink()


def _existing_complete_shards(out_dir: Path, split_tag: str, samples_per_shard: int) -> tuple[int, int | None]:
    bins: list[tuple[int, int, str]] = []
    if not out_dir.is_dir():
        return 0, None
    for filename in os.listdir(out_dir):
        parsed = _parse_shard_name(filename, split_tag, "gdvlm")
        if parsed:
            bins.append((parsed[0] - 1, parsed[1], filename))
    if not bins:
        return 0, None

    totals = {total for _, total, _ in bins}
    file_num_shards = max(totals)
    if len(totals) > 1:
        print(f"warning: mixed shard totals; resuming of-{file_num_shards:05d}")
    bins = sorted((idx, name) for idx, total, name in bins if total == file_num_shards)
    by_idx = {idx: name for idx, name in bins}

    complete = 0
    for idx in range(len(bins)):
        if idx not in by_idx:
            break
        idx_path = _shard_path(out_dir, split_tag, idx, file_num_shards, "idx")
        if not idx_path.is_file():
            break
        if _idx_entry_count(idx_path) != samples_per_shard:
            break
        complete += 1

    # Recompute first partial shard and anything after it.
    for idx, _name in bins:
        if idx < complete:
            continue
        for ext in ("gdvlm", "idx"):
            path = _shard_path(out_dir, split_tag, idx, file_num_shards, ext)
            if path.exists():
                path.unlink()
    return complete, file_num_shards


def write_shard(
    path: Path,
    blobs: list[bytes],
    shard_idx: int,
    num_shards: int,
    token_dtype: int,
    vocab_size: int,
    tokenizer_hash_u64: int,
) -> None:
    header = SHARD_HEADER_STRUCT.pack(
        SHARD_MAGIC,
        SHARD_VERSION,
        SHARD_HEADER_SIZE,
        TARGET_H,
        TARGET_W,
        CHANNELS,
        PATCH_SIZE,
        NUM_PATCHES,
        PATCH_DIM,
        PATCH_DTYPE_F16,
        token_dtype,
        vocab_size,
        shard_idx,
        num_shards,
        SHARD_HEADER_SIZE,
        len(blobs),
        tokenizer_hash_u64,
    )
    if len(header) > SHARD_HEADER_SIZE:
        raise AssertionError("shard header exceeds fixed size")

    # Shards are ~600 MiB at the default 2048 samples/shard. Atomic temp-file
    # writes can double peak disk use when rerunning over stale outputs, so write
    # the large shard directly. Resume safety comes from write order: shard first,
    # idx second. A crash during shard write leaves no complete idx, so next run
    # deletes/recomputes the shard.
    if path.exists():
        path.unlink()
    try:
        with path.open("wb") as f:
            f.write(header)
            f.write(b"\0" * (SHARD_HEADER_SIZE - len(header)))
            for blob in blobs:
                f.write(blob)
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_idx(path: Path, entries: list[bytes]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        for entry in entries:
            f.write(entry)
    os.replace(tmp, path)


def _flush(
    blobs: list[bytes],
    entries: list[bytes],
    shard_idx: int,
    num_shards: int,
    out_dir: Path,
    split_tag: str,
    token_dtype: int,
    vocab_size: int,
    tokenizer_hash_u64: int,
) -> None:
    bin_path = _shard_path(out_dir, split_tag, shard_idx, num_shards, "gdvlm")
    idx_path = _shard_path(out_dir, split_tag, shard_idx, num_shards, "idx")
    write_shard(bin_path, blobs, shard_idx, num_shards, token_dtype, vocab_size, tokenizer_hash_u64)
    write_idx(idx_path, entries)
    display_total = max(num_shards, shard_idx + 1)
    print(f"  wrote shard {shard_idx + 1}/{display_total} samples={len(blobs)} path={bin_path.name}")


def _finalize_shard_count(out_dir: Path, split_tag: str, old_num_shards: int, actual_num_shards: int) -> None:
    if actual_num_shards == old_num_shards:
        return
    for shard_idx in range(actual_num_shards):
        for ext in ("gdvlm", "idx"):
            old = _shard_path(out_dir, split_tag, shard_idx, old_num_shards, ext)
            new = _shard_path(out_dir, split_tag, shard_idx, actual_num_shards, ext)
            if old != new and old.exists():
                os.replace(old, new)
        bin_path = _shard_path(out_dir, split_tag, shard_idx, actual_num_shards, "gdvlm")
        if bin_path.exists():
            with bin_path.open("r+b") as f:
                f.seek(SHARD_NUM_SHARDS_OFFSET)
                f.write(struct.pack("<I", actual_num_shards))


def _write_combined_idx(out_dir: Path, split_tag: str, actual_num_shards: int) -> int:
    split_idx_path = out_dir / f"{split_tag}.idx"
    tmp = split_idx_path.with_suffix(split_idx_path.suffix + ".tmp")
    total_entries = 0
    with tmp.open("wb") as out:
        out.write(IDX_HEADER_STRUCT.pack(IDX_MAGIC, IDX_VERSION, IDX_ENTRY_BYTES, 0))
        for shard_idx in range(actual_num_shards):
            p = _shard_path(out_dir, split_tag, shard_idx, actual_num_shards, "idx")
            if not p.exists():
                continue
            data = p.read_bytes()
            if len(data) % IDX_ENTRY_BYTES != 0:
                raise ValueError(f"bad idx size: {p}")
            total_entries += len(data) // IDX_ENTRY_BYTES
            out.write(data)
        out.seek(0)
        out.write(IDX_HEADER_STRUCT.pack(IDX_MAGIC, IDX_VERSION, IDX_ENTRY_BYTES, total_entries))
    os.replace(tmp, split_idx_path)
    return total_entries


def _combined_idx_token_stats(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < IDX_HEADER_STRUCT.size:
        raise ValueError(f"combined idx too small: {path}")
    magic, version, entry_size, n_entries = IDX_HEADER_STRUCT.unpack_from(data, 0)
    if magic != IDX_MAGIC or version != IDX_VERSION or entry_size != IDX_ENTRY_BYTES:
        raise ValueError(f"bad combined idx header: {path}")
    expected = IDX_HEADER_STRUCT.size + int(n_entries) * IDX_ENTRY_BYTES
    if len(data) != expected:
        raise ValueError(f"combined idx size mismatch: {path}")
    text_tokens = 0
    offset = IDX_HEADER_STRUCT.size
    for _ in range(int(n_entries)):
        entry = IDX_ENTRY_STRUCT.unpack_from(data, offset)
        text_tokens += int(entry[4])
        offset += IDX_ENTRY_BYTES
    return int(n_entries), text_tokens


def _write_meta(out_dir: Path, split_tag: str, meta: dict[str, Any]) -> None:
    path = _meta_path(out_dir, split_tag)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_dataset_config(out_dir: Path, tokenized_path: Path, tokenized: dict[str, Any], token_dtype: int) -> None:
    path = out_dir / "dataset_config.json"
    payload = {
        "format": "gd-imagenet-vlm-dataset-config-v1",
        "image": {
            "height": TARGET_H,
            "width": TARGET_W,
            "channels": CHANNELS,
            "resize_short": RESIZE_SHORT,
            "patch_size": PATCH_SIZE,
            "num_patches": NUM_PATCHES,
            "patch_dim": PATCH_DIM,
            "patch_dtype": "f16",
            "patch_bytes_per_sample": PATCH_BYTES_PER_SAMPLE,
            "normalization": {
                "mean": IMAGENET_MEAN.tolist(),
                "std": IMAGENET_STD.tolist(),
            },
            "patch_order": "CHW within each patch; patch grid row-major",
        },
        "text": {
            "tokenized_text_path": str(tokenized_path),
            "tokenizer_path": tokenized["tokenizer_path"],
            "tokenizer_hash": tokenized["tokenizer_hash"],
            "vocab_size": tokenized["vocab_size"],
            "token_dtype": _token_dtype_name(token_dtype),
            "pad_id": tokenized["pad_id"],
            "loss": "mask image prefix; train suffix text tokens only",
        },
        "sequence": {
            "image_prefix_tokens": NUM_PATCHES,
            "attention": {
                "causal": True,
                "prefix_len": NUM_PATCHES,
                "sliding_window_default": 128,
            },
        },
        "binary_format": {
            "shard_magic": SHARD_MAGIC.decode("ascii", errors="replace").rstrip("\0"),
            "shard_header_bytes": SHARD_HEADER_SIZE,
            "idx_magic": IDX_MAGIC.decode("ascii"),
            "idx_entry_bytes": IDX_ENTRY_BYTES,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def process_split(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    tokenized_path = Path(args.tokenized_text)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenized = load_tokenized_text(tokenized_path)
    token_dtype = _token_dtype_for_vocab(tokenized["vocab_size"])
    token_size = _token_dtype_size(token_dtype)
    _write_dataset_config(out_dir, tokenized_path, tokenized, token_dtype)

    split_tag = "train" if args.split == "train" else "val"
    data_dir = in_dir / "data"
    parquet_files = sorted(p for p in data_dir.iterdir() if p.name.startswith(args.split) and p.suffix == ".parquet")
    if not parquet_files:
        raise FileNotFoundError(f"no parquet shards found for split '{args.split}' in {data_dir}")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    total_rows = 0
    for parquet_file in parquet_files:
        total_rows += pq.ParquetFile(parquet_file).metadata.num_rows
    row_limit = min(args.limit, total_rows) if args.limit > 0 else total_rows
    planned_shards = math.ceil(row_limit / args.samples_per_shard)

    print(f"input: {in_dir}")
    print(f"output: {out_dir}")
    print(f"split: source={args.split} tag={split_tag}")
    print(f"parquet shards: {len(parquet_files)} rows={total_rows} limit={row_limit}")
    print(
        "image: "
        f"{TARGET_H}x{TARGET_W} patch={PATCH_SIZE} patches={NUM_PATCHES} "
        f"patch_dim={PATCH_DIM} dtype=f16 bytes/sample={PATCH_BYTES_PER_SAMPLE}"
    )
    print(
        "text: "
        f"labels={len(tokenized['by_id'])} vocab={tokenized['vocab_size']} "
        f"token_dtype={_token_dtype_name(token_dtype)} tokenizer_hash={tokenized['tokenizer_hash']}"
    )
    print(f"planned output shards: up to {planned_shards} samples_per_shard={args.samples_per_shard}")

    _clean_stale_tmp_outputs(out_dir, split_tag)

    shard_raw_ends: list[int] = []
    if args.no_resume:
        _clean_split_outputs(out_dir, split_tag)
        out_idx = 0
        file_num_shards = planned_shards
    else:
        out_idx, existing_num_shards = _existing_complete_shards(out_dir, split_tag, args.samples_per_shard)
        file_num_shards = existing_num_shards or planned_shards
        meta = _read_meta(out_dir, split_tag)
        if meta and meta.get("samples_per_shard") == args.samples_per_shard:
            shard_raw_ends = list(meta.get("shard_raw_ends", []))[:out_idx]
        if existing_num_shards and existing_num_shards != planned_shards:
            print(f"resume: keeping filename count of-{existing_num_shards:05d} (planned {planned_shards})")
        if out_idx:
            print(f"resume: {out_idx} complete shards exist")

    resume_rows = shard_raw_ends[-1] if shard_raw_ends else out_idx * args.samples_per_shard
    total_seen = 0
    skipped = 0
    blobs: list[bytes] = []
    entries: list[bytes] = []
    body_offset = 0
    processed_samples = 0
    image_tokens = 0
    text_tokens = 0
    bad_label_counts: dict[int, int] = {}

    with Pool(args.num_workers) as pool:
        for parquet_file in tqdm(parquet_files, desc="Parquet"):
            if total_seen >= row_limit:
                break
            pf = pq.ParquetFile(parquet_file)
            n_in_shard = pf.metadata.num_rows
            rows_left = row_limit - total_seen
            rows_this_shard = min(n_in_shard, rows_left)
            if total_seen + rows_this_shard <= resume_rows:
                total_seen += rows_this_shard
                continue

            table = pq.read_table(parquet_file, columns=["image", "label"])
            img_col = _image_bytes_pylist(table)
            label_col = table.column("label").to_pylist()
            del table
            gc.collect()

            batch_args: list[tuple[int, bytes, int, list[int], int]] = []
            for i in range(n_in_shard):
                if total_seen >= row_limit:
                    break
                if total_seen < resume_rows:
                    total_seen += 1
                    continue
                img_raw = _coerce_image_bytes(img_col[i])
                label_id = int(label_col[i])
                total_seen += 1
                if img_raw is None:
                    skipped += 1
                    continue
                label_entry = tokenized["by_id"].get(label_id)
                if label_entry is None:
                    bad_label_counts[label_id] = bad_label_counts.get(label_id, 0) + 1
                    skipped += 1
                    continue
                batch_args.append((total_seen, img_raw, label_id, label_entry["ids"], token_dtype))

            if not batch_args:
                continue
            chunksize = max(1, len(batch_args) // max(1, args.num_workers * 8))
            for result in tqdm(
                pool.imap(process_sample, batch_args, chunksize=chunksize),
                total=len(batch_args),
                desc=f"  {parquet_file.name}",
                leave=False,
            ):
                raw_pos = int(result[0])
                if result[1] is None:
                    skipped += 1
                    continue
                _raw_pos, blob, label_id, token_len = result
                record_nbytes = len(blob)
                sample_idx = len(blobs)
                if token_len > 0xFFFFFFFF or record_nbytes > 0xFFFFFFFF:
                    raise ValueError("record metadata exceeds u32")
                entries.append(
                    IDX_ENTRY_STRUCT.pack(
                        out_idx,
                        sample_idx,
                        body_offset,
                        record_nbytes,
                        token_len,
                        int(label_id) & 0xFFFF,
                        0,
                        raw_pos & 0xFFFFFFFF,
                    )
                )
                body_offset += record_nbytes
                blobs.append(blob)
                processed_samples += 1
                image_tokens += NUM_PATCHES
                text_tokens += token_len

                if len(blobs) >= args.samples_per_shard:
                    _flush(
                        blobs,
                        entries,
                        out_idx,
                        file_num_shards,
                        out_dir,
                        split_tag,
                        token_dtype,
                        tokenized["vocab_size"],
                        tokenized["tokenizer_hash_u64"],
                    )
                    shard_raw_ends.append(raw_pos)
                    out_idx += 1
                    blobs = []
                    entries = []
                    body_offset = 0

    if blobs:
        _flush(
            blobs,
            entries,
            out_idx,
            file_num_shards,
            out_dir,
            split_tag,
            token_dtype,
            tokenized["vocab_size"],
            tokenized["tokenizer_hash_u64"],
        )
        shard_raw_ends.append(total_seen)
        out_idx += 1

    actual_num_shards = out_idx
    if actual_num_shards <= 0:
        raise RuntimeError("no output shards were produced")
    _finalize_shard_count(out_dir, split_tag, file_num_shards, actual_num_shards)
    total_in_idx = _write_combined_idx(out_dir, split_tag, actual_num_shards)
    idx_samples, idx_text_tokens = _combined_idx_token_stats(out_dir / f"{split_tag}.idx")
    if idx_samples != total_in_idx:
        raise ValueError("combined idx entry count mismatch")

    skipped_total = max(0, total_seen - total_in_idx)
    idx_image_tokens = idx_samples * NUM_PATCHES
    idx_total_tokens = idx_image_tokens + idx_text_tokens
    idx_avg_text = idx_text_tokens / idx_samples if idx_samples else 0.0
    idx_avg_total = idx_total_tokens / idx_samples if idx_samples else 0.0
    total_tokens = image_tokens + text_tokens
    avg_text = text_tokens / processed_samples if processed_samples else 0.0
    avg_total = total_tokens / processed_samples if processed_samples else 0.0
    meta = {
        "format": "gd-imagenet-vlm-split-meta-v1",
        "split": split_tag,
        "source_split": args.split,
        "source_dir": str(in_dir),
        "row_limit": row_limit,
        "raw_seen": total_seen,
        "samples": idx_samples,
        "samples_this_run": processed_samples,
        "skipped": skipped_total,
        "bad_label_counts": bad_label_counts,
        "samples_per_shard": args.samples_per_shard,
        "num_shards": actual_num_shards,
        "shard_raw_ends": shard_raw_ends,
        "image": {
            "height": TARGET_H,
            "width": TARGET_W,
            "channels": CHANNELS,
            "patch_size": PATCH_SIZE,
            "num_patches": NUM_PATCHES,
            "patch_dim": PATCH_DIM,
            "patch_dtype": "f16",
            "patch_bytes_per_sample": PATCH_BYTES_PER_SAMPLE,
        },
        "text": {
            "tokenized_text_path": str(tokenized_path),
            "vocab_size": tokenized["vocab_size"],
            "token_dtype": _token_dtype_name(token_dtype),
            "token_dtype_bytes": token_size,
            "tokenizer_hash": tokenized["tokenizer_hash"],
            "pad_id": tokenized["pad_id"],
        },
        "tokens": {
            "image_tokens": idx_image_tokens,
            "text_tokens": idx_text_tokens,
            "total_tokens": idx_total_tokens,
            "avg_text_tokens_per_sample": idx_avg_text,
            "avg_total_tokens_per_sample": idx_avg_total,
        },
        "tokens_this_run": {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "total_tokens": total_tokens,
            "avg_text_tokens_per_sample": avg_text,
            "avg_total_tokens_per_sample": avg_total,
        },
    }
    _write_meta(out_dir, split_tag, meta)

    print(f"wrote combined idx: {out_dir / (split_tag + '.idx')} entries={total_in_idx}")
    print(f"wrote meta: {_meta_path(out_dir, split_tag)}")
    print(
        "tokens complete split: "
        f"samples={idx_samples} image_tokens={idx_image_tokens} "
        f"text_tokens={idx_text_tokens} total_tokens={idx_total_tokens} "
        f"avg_text={idx_avg_text:.2f} avg_total={idx_avg_total:.2f}"
    )
    print(
        "tokens this run: "
        f"samples={processed_samples} image_tokens={image_tokens} "
        f"text_tokens={text_tokens} total_tokens={total_tokens} "
        f"avg_text={avg_text:.2f} avg_total={avg_total:.2f}"
    )
    print(f"done: split={split_tag} samples_in_idx={total_in_idx} shards={actual_num_shards} skipped={skipped_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ImageNet VLM patches into gradients.c shards")
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--tokenized-text", default=str(Path(DEFAULT_OUT_DIR) / "text-tokenized.json"))
    parser.add_argument("--samples-per-shard", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="max raw rows to process (0 = all)")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--no-resume", action="store_true", help="delete existing outputs for this split first")
    args = parser.parse_args()
    process_split(args)


if __name__ == "__main__":
    main()
