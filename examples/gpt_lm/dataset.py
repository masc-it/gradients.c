#!/usr/bin/env python3
"""Prepare a tiny packed GPT language-modeling GDDS dataset.

The default source is ``~/projects/dnn.c/docs/promessi_sposi.txt``.  The script:

1. splits raw text into deterministic block-random train/validation files;
2. trains a byte-level BPE tokenizer with a 2048 token vocabulary on the raw
   training text only;
3. tokenizes train and validation text with that train-fitted tokenizer into
   packed records of ``context_length + 1`` tokens with one-token overlap for
   next-token prediction;
4. writes GDDS train/val shards under ``examples/gpt_lm/data`` with shifted
   ``input_ids``/``target_ids`` tensors of length ``context_length``; and
5. prints decoded random samples loaded back from the produced GDDS shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gdds_utils import (  # noqa: E402
    FieldSpec,
    GddsSplitWriter,
    TensorData,
    field_metadata,
    schema_hash,
)
from tok_utils import TOKEN_DTYPE, TokenizedCorpus, tokenize_file_packed  # noqa: E402


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_CONTEXT_LENGTH = 512
MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
DATA_FORMAT_VERSION = "gpt-lm-promessi-v3"
DEFAULT_SPLIT_BLOCK_CHARS = 16 * 1024
DEFAULT_CHARS_PER_TOKEN_ESTIMATE = 3.0
TOKENIZER_CACHE_FORMAT = "gpt-lm-tokenizer-cache-v1"


@dataclass(frozen=True)
class ShardHeader:
    samples: int
    index_offset: int
    schema_hash: int


@dataclass(frozen=True)
class RawTextSplit:
    train_path: Path
    val_path: Path
    total_blocks: int
    train_blocks: int
    val_blocks: int
    total_chars: int
    train_chars: int
    val_chars: int
    target_val_chars: int
    train_sha256: str
    val_sha256: str


class TokenDecoder:
    """Small byte-token decoder for previewing generated samples."""

    def __init__(self, tokenizer_path: Path):
        with tokenizer_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
        self._pieces: dict[int, bytes] = {}
        for token in spec["tokens"]:
            token_id = int(token["id"])
            if token["kind"] == "special":
                self._pieces[token_id] = str(token["text"]).encode("utf-8")
            else:
                self._pieces[token_id] = bytes.fromhex(str(token["hex"]))

    def decode(self, token_ids: Sequence[int] | np.ndarray) -> str:
        data = bytearray()
        for token_id in token_ids:
            data.extend(self._pieces[int(token_id)])
        return bytes(data).decode("utf-8", errors="replace")


def fields_for_context(context_length: int) -> list[FieldSpec]:
    # Store fixed-size per-sample arrays but collate them as packed sequences.
    # The dataloader then exposes:
    #   input_ids:  [B * context_length]
    #   target_ids: [B * context_length]
    #   cu_seqlens: [B + 1]
    #   positions:  [B * context_length]
    # which is exactly what RoPE + sdpa_varlen want.
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return [
        FieldSpec("input_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec(
            "cu_seqlens",
            "i32",
            (-1,),
            collate="generated",
            generated="cu_seqlens",
            source="input_ids",
        ),
        FieldSpec(
            "positions",
            "i32",
            (-1,),
            collate="generated",
            generated="positions",
            source="input_ids",
        ),
        FieldSpec("target_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
    ]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_split_shards(out_dir: Path, split: str) -> None:
    for path in out_dir.glob(f"{split}-*.gdds"):
        path.unlink()


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_cache_path(tokenizer_path: Path) -> Path:
    return tokenizer_path.with_name(f"{tokenizer_path.name}.train-cache.json")


def split_text_into_blocks(text: str, block_chars: int) -> list[str]:
    """Split text into deterministic, roughly equal contiguous character blocks.

    The split prefers paragraph boundaries near the target size, then line breaks,
    then whitespace.  We choose validation blocks from these raw-text blocks
    before tokenizer training, which avoids both tail-only validation and BPE
    leakage from validation text.
    """
    if block_chars <= 0:
        raise ValueError("block_chars must be positive")
    n_chars = len(text)
    blocks: list[str] = []
    start = 0
    while start < n_chars:
        remaining = n_chars - start
        if remaining <= int(block_chars * 1.25):
            end = n_chars
        else:
            target = start + block_chars
            lo = min(n_chars, start + max(1, block_chars // 2))
            hi = min(n_chars, start + block_chars + max(1, block_chars // 2))
            end = _best_text_boundary(text, lo, hi, target)
            if end <= start:
                end = min(n_chars, start + block_chars)
        block = text[start:end]
        if block.strip():
            blocks.append(block)
        start = end
    if not blocks:
        raise ValueError("source text produced no non-empty split blocks")
    return blocks


def _best_text_boundary(text: str, lo: int, hi: int, target: int) -> int:
    boundary_patterns = [r"\n\s*\n+", r"\n", r"\s+"]
    for pattern in boundary_patterns:
        candidates = [lo + m.end() for m in re.finditer(pattern, text[lo:hi])]
        if candidates:
            return min(candidates, key=lambda pos: abs(pos - target))
    return hi


def choose_validation_blocks(
    blocks: Sequence[str],
    *,
    val_samples: int,
    context_length: int,
    val_fraction: float | None,
    split_seed: int,
    chars_per_token_estimate: float,
) -> tuple[set[int], int]:
    if val_samples < 0:
        raise ValueError("val_samples must be non-negative")
    if context_length < 2:
        raise ValueError("context_length must be >= 2")
    if chars_per_token_estimate <= 0.0:
        raise ValueError("chars_per_token_estimate must be positive")
    total_chars = sum(len(block) for block in blocks)
    if len(blocks) <= 1 or val_samples == 0:
        return set(), 0
    if val_fraction is not None:
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
        target_val_chars = int(round(total_chars * val_fraction))
    else:
        target_val_chars = int(round(val_samples * (context_length + 1) * chars_per_token_estimate))
    if target_val_chars <= 0:
        return set(), 0
    # Keep at least one block for training.  This also prevents accidentally
    # validating on most of a tiny source file when --val-samples is large.
    max_val_chars = max(1, total_chars - min(len(block) for block in blocks))
    target_val_chars = min(target_val_chars, max_val_chars)

    order = list(range(len(blocks)))
    random.Random(split_seed).shuffle(order)
    selected: set[int] = set()
    selected_chars = 0
    for index in order:
        if len(selected) >= len(blocks) - 1:
            break
        selected.add(index)
        selected_chars += len(blocks[index])
        if selected_chars >= target_val_chars:
            break
    return selected, target_val_chars


def write_raw_text_split(
    *,
    source: Path,
    data_dir: Path,
    context_length: int,
    val_samples: int,
    val_fraction: float | None,
    split_seed: int,
    split_block_chars: int,
    chars_per_token_estimate: float,
) -> RawTextSplit:
    text = source.read_text(encoding="utf-8")
    blocks = split_text_into_blocks(text, split_block_chars)
    val_indices, target_val_chars = choose_validation_blocks(
        blocks,
        val_samples=val_samples,
        context_length=context_length,
        val_fraction=val_fraction,
        split_seed=split_seed,
        chars_per_token_estimate=chars_per_token_estimate,
    )
    split_dir = data_dir / "raw_split"
    ensure_clean_dir(split_dir)
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    train_blocks = [block for i, block in enumerate(blocks) if i not in val_indices]
    val_blocks = [block for i, block in enumerate(blocks) if i in val_indices]
    if not train_blocks:
        raise ValueError("raw split produced no training blocks")
    _write_text_blocks(train_path, train_blocks)
    _write_text_blocks(val_path, val_blocks)
    train_sha = file_sha256(train_path)
    val_sha = file_sha256(val_path) if val_blocks else ""
    train_chars = sum(len(block) for block in train_blocks)
    val_chars = sum(len(block) for block in val_blocks)
    print(
        "Raw split: "
        f"blocks={len(blocks)} train_blocks={len(train_blocks)} val_blocks={len(val_blocks)} "
        f"train_chars={train_chars} val_chars={val_chars} target_val_chars={target_val_chars} "
        f"seed={split_seed}"
    )
    return RawTextSplit(
        train_path=train_path,
        val_path=val_path,
        total_blocks=len(blocks),
        train_blocks=len(train_blocks),
        val_blocks=len(val_blocks),
        total_chars=len(text),
        train_chars=train_chars,
        val_chars=val_chars,
        target_val_chars=target_val_chars,
        train_sha256=train_sha,
        val_sha256=val_sha,
    )


def _write_text_blocks(path: Path, blocks: Sequence[str]) -> None:
    # Preserve source order within each split.  Separators avoid fusing words
    # across held-out gaps in the training text.
    text = "\n\n".join(block.strip("\n") for block in blocks if block.strip())
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def tokenizer_compatible(tokenizer_path: Path, *, vocab_size: int, specials: Sequence[str]) -> bool:
    try:
        spec = read_json(tokenizer_path)
        if int(spec.get("vocab_size", -1)) != vocab_size:
            return False
        present = {
            str(token["text"])
            for token in spec.get("tokens", [])
            if isinstance(token, dict) and token.get("kind") == "special"
        }
        return all(token in present for token in specials)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def tokenizer_cache_compatible(
    tokenizer_path: Path,
    *,
    train_sha256: str,
    vocab_size: int,
    specials: Sequence[str],
) -> bool:
    if not tokenizer_compatible(tokenizer_path, vocab_size=vocab_size, specials=specials):
        return False
    try:
        cache = read_json(tokenizer_cache_path(tokenizer_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        cache.get("format") == TOKENIZER_CACHE_FORMAT
        and cache.get("train_sha256") == train_sha256
        and int(cache.get("vocab_size", -1)) == vocab_size
        and list(cache.get("specials", [])) == list(specials)
    )


def write_tokenizer_cache(
    tokenizer_path: Path,
    *,
    train_sha256: str,
    vocab_size: int,
    specials: Sequence[str],
) -> None:
    cache = {
        "format": TOKENIZER_CACHE_FORMAT,
        "train_sha256": train_sha256,
        "vocab_size": vocab_size,
        "specials": list(specials),
        "note": "Tokenizer was trained only on raw_split/train.txt.",
    }
    tokenizer_cache_path(tokenizer_path).write_text(json.dumps(cache, indent=2) + "\n")


def prepare_tokenized_corpus(
    *,
    split_name: str,
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    context_length: int,
    vocab_size: int,
    use_existing_tokenizer: bool,
) -> TokenizedCorpus:
    ensure_clean_dir(output_dir)
    summary = tokenize_file_packed(
        tokenizer=tokenizer_path,
        input_path=input_path,
        output_dir=output_dir,
        im_start=IM_START,
        im_end=IM_END,
        num_tokens_per_sequence=context_length + 1,
        use_tokenizer=use_existing_tokenizer,
        vocab_size=vocab_size,
        min_frequency=2,
        seed=17,
    )
    action = "tokenized with existing tokenizer" if use_existing_tokenizer else "trained tokenizer"
    print(
        f"{split_name}: {action}: {summary['tokenizer']} "
        f"(records={summary['sequences']}, record_tokens={context_length + 1}, "
        f"tokens={summary['tokens']})"
    )
    return TokenizedCorpus.open(output_dir)


def i32_tensor(array: np.ndarray) -> TensorData:
    arr = np.asarray(array, dtype=TOKEN_DTYPE)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return TensorData(dtype="i32", shape=(int(arr.size),), data=arr.tobytes(order="C"))


def iter_lm_samples(corpus: TokenizedCorpus):
    for packed in corpus.iter_packed_arrays():
        if packed.shape[1] != corpus.sequence_length:
            raise ValueError(f"unexpected packed shard width {packed.shape[1]}")
        for row in packed:
            yield {
                "input_ids": i32_tensor(row[:-1]),
                "target_ids": i32_tensor(row[1:]),
            }


def write_gdds_dataset(
    *,
    out_dir: Path,
    train_corpus: TokenizedCorpus,
    val_corpus: TokenizedCorpus | None,
    max_shard_bytes: int,
) -> dict[str, list[Path]]:
    if train_corpus.sequence_length is None or train_corpus.sequence_length < 3:
        raise ValueError("packed training corpus must have sequence_length >= 3")
    if val_corpus is not None and val_corpus.sequence_length != train_corpus.sequence_length:
        raise ValueError("train and validation corpora must use the same packed sequence length")
    context_length = train_corpus.sequence_length - 1
    fields = fields_for_context(context_length)
    if train_corpus.num_sequences <= 0:
        raise ValueError("tokenizer produced no training records")
    remove_split_shards(out_dir, "train")
    remove_split_shards(out_dir, "val")
    train_writer = GddsSplitWriter(out_dir, "train", fields, max_shard_bytes=max_shard_bytes)
    val_writer = GddsSplitWriter(out_dir, "val", fields, max_shard_bytes=max_shard_bytes)
    try:
        for sample in iter_lm_samples(train_corpus):
            train_writer.write_sample(sample)
        if val_corpus is not None:
            for sample in iter_lm_samples(val_corpus):
                val_writer.write_sample(sample)
        train_paths = train_writer.finish()
        val_paths = val_writer.finish()
    except BaseException:
        train_writer.abort()
        val_writer.abort()
        raise
    if not train_paths:
        raise ValueError("tokenizer produced no training records")
    return {"train": train_paths, "val": val_paths}


def read_gdds_header(path: Path) -> ShardHeader:
    with path.open("rb") as f:
        header = f.read(128)
    if len(header) != 128 or header[:8] != b"GDDSv1\0\0":
        raise ValueError(f"{path} is not a GDDS v1 shard")
    return ShardHeader(
        samples=struct.unpack_from("<Q", header, 24)[0],
        index_offset=struct.unpack_from("<Q", header, 40)[0],
        schema_hash=struct.unpack_from("<Q", header, 64)[0],
    )


def write_manifest(
    *,
    out_dir: Path,
    source: Path,
    tokenizer_path: Path,
    raw_split: RawTextSplit,
    train_corpus: TokenizedCorpus,
    val_corpus: TokenizedCorpus | None,
    gdds_paths: Mapping[str, Sequence[Path]],
    max_shard_bytes: int,
    vocab_size: int,
    split_seed: int,
    split_block_chars: int,
    val_samples_target: int,
    val_fraction: float | None,
) -> None:
    assert train_corpus.sequence_length is not None
    context_length = train_corpus.sequence_length - 1
    fields = fields_for_context(context_length)
    split_samples = {
        split: sum(read_gdds_header(path).samples for path in paths)
        for split, paths in gdds_paths.items()
    }
    val_records = 0 if val_corpus is None else val_corpus.num_sequences
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    manifest = {
        "format": "GDDS",
        "version": 1,
        "schema_hash": f"0x{schema_hash(fields):016x}",
        "fields": [field_metadata(field) for field in fields],
        "storage": {
            "input_ids": f"per-sample int32 token ids, length {context_length}; batch-collated packed",
            "target_ids": "input_ids shifted by one token for next-token prediction; batch-collated packed",
            "cu_seqlens": "generated by GDDS dataloader from input_ids; shape [batch + 1]",
            "positions": "generated by GDDS dataloader from input_ids; shape [sum sequence lengths]",
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "hash": train_corpus.manifest["tokenizer_hash"],
            "vocab_size": vocab_size,
            "im_start_id": train_corpus.im_start_id,
            "im_end_id": train_corpus.im_end_id,
            "digits": "always split before BPE merges",
            "trained_on": "raw_split/train.txt",
            "train_sha256": raw_split.train_sha256,
        },
        "packing": {
            "record_length": train_corpus.sequence_length,
            "context_length": context_length,
            "stride": train_corpus.stride,
            "source_records": train_corpus.num_sequences + val_records,
            "source_tokens": train_corpus.num_tokens + val_tokens,
            "train_records": train_corpus.num_sequences,
            "val_records": val_records,
        },
        "splits": {
            split: {
                "samples": split_samples.get(split, 0),
                "shards": [path.name for path in paths],
            }
            for split, paths in gdds_paths.items()
        },
        "raw_split": {
            "mode": "block_random_before_tokenizer_training",
            "train_path": str(raw_split.train_path.relative_to(out_dir)),
            "val_path": str(raw_split.val_path.relative_to(out_dir)),
            "split_seed": split_seed,
            "block_chars": split_block_chars,
            "total_blocks": raw_split.total_blocks,
            "train_blocks": raw_split.train_blocks,
            "val_blocks": raw_split.val_blocks,
            "total_chars": raw_split.total_chars,
            "train_chars": raw_split.train_chars,
            "val_chars": raw_split.val_chars,
            "target_val_chars": raw_split.target_val_chars,
            "target_val_samples": val_samples_target,
            "val_fraction": val_fraction,
            "train_sha256": raw_split.train_sha256,
            "val_sha256": raw_split.val_sha256,
        },
        "prep": {
            "format_version": DATA_FORMAT_VERSION,
            "source": str(source),
            "max_shard_bytes": max_shard_bytes,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_gdds_lm_sample(shards: Sequence[Path], sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    remaining = sample_index
    for path in shards:
        header = read_gdds_header(path)
        if remaining >= header.samples:
            remaining -= header.samples
            continue
        with path.open("rb") as f:
            f.seek(header.index_offset + remaining * 16)
            index_entry = f.read(16)
            if len(index_entry) != 16:
                raise ValueError(f"truncated GDDS index in {path}")
            record_offset, record_nbytes = struct.unpack("<QQ", index_entry)
            f.seek(record_offset)
            record = f.read(record_nbytes)
        if len(record) != record_nbytes or record[:4] != b"GDDR":
            raise ValueError(f"invalid GDDS record in {path}")
        n_entries = struct.unpack_from("<H", record, 4)[0]
        header_nbytes = struct.unpack_from("<I", record, 8)[0]
        arrays: dict[int, np.ndarray] = {}
        for entry_index in range(n_entries):
            base = 20 + entry_index * 88
            field_id, rank, _flags = struct.unpack_from("<HHI", record, base)
            dims = struct.unpack_from("<8q", record, base + 8)[:rank]
            payload_offset, payload_nbytes = struct.unpack_from("<QQ", record, base + 72)
            data = record[header_nbytes + payload_offset : header_nbytes + payload_offset + payload_nbytes]
            arrays[field_id] = np.frombuffer(data, dtype=TOKEN_DTYPE).reshape(dims).copy()
        if 0 not in arrays:
            raise ValueError(f"missing input_ids in {path}")
        target_field_id = max(arrays)
        if target_field_id == 0:
            raise ValueError(f"missing target_ids in {path}")
        return arrays[0], arrays[target_field_id]
    raise IndexError(sample_index)


def show_random_samples(
    *,
    shards: Sequence[Path],
    tokenizer_path: Path,
    total_samples: int,
    count: int,
    seed: int,
) -> None:
    decoder = TokenDecoder(tokenizer_path)
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_samples), k=min(count, total_samples)))
    print(f"\nDecoded random samples ({len(selected)}):")
    for sample_index in selected:
        input_ids, target_ids = read_gdds_lm_sample(shards, sample_index)
        if not np.array_equal(input_ids[1:], target_ids[:-1]):
            raise ValueError(f"sample {sample_index} is not correctly shifted")
        full = np.empty(input_ids.shape[0] + 1, dtype=TOKEN_DTYPE)
        full[:-1] = input_ids
        full[-1] = target_ids[-1]
        text = decoder.decode(full)
        print(f"\n--- sample {sample_index} ({len(full)} tokens) ---")
        print(text[:900].replace("\0", "�"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "projects/dnn.c/docs/promessi_sposi.txt",
        help="input text corpus",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="model input length; stored tokenized records contain context_length + 1 tokens",
    )
    parser.add_argument(
        "--sequence-length",
        dest="context_length",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument(
        "--val-samples",
        type=int,
        default=64,
        help="approximate validation budget in packed samples; converted to a raw-text char budget before tokenization",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="override --val-samples and reserve this raw-text fraction for validation",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--split-block-chars", type=int, default=DEFAULT_SPLIT_BLOCK_CHARS)
    parser.add_argument("--chars-per-token-estimate", type=float, default=DEFAULT_CHARS_PER_TOKEN_ESTIMATE)
    parser.add_argument("--preview-samples", type=int, default=10)
    parser.add_argument("--preview-seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if args.vocab_size < 258:
        raise ValueError("--vocab-size must be at least 258 for byte tokens plus two specials")
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")
    if args.val_samples < 0:
        raise ValueError("--val-samples must be non-negative")
    if args.val_fraction is not None and not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1")
    if args.split_block_chars <= 0:
        raise ValueError("--split-block-chars must be positive")
    if args.chars_per_token_estimate <= 0.0:
        raise ValueError("--chars-per-token-estimate must be positive")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / f"tokenizer-v{args.vocab_size}.json"

    raw_split = write_raw_text_split(
        source=args.source,
        data_dir=out_dir,
        context_length=args.context_length,
        val_samples=args.val_samples,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        split_block_chars=args.split_block_chars,
        chars_per_token_estimate=args.chars_per_token_estimate,
    )
    specials = (IM_START, IM_END)
    reuse_train_tokenizer = (
        not args.retrain_tokenizer
        and tokenizer_cache_compatible(
            tokenizer_path,
            train_sha256=raw_split.train_sha256,
            vocab_size=args.vocab_size,
            specials=specials,
        )
    )
    train_corpus = prepare_tokenized_corpus(
        split_name="train",
        input_path=raw_split.train_path,
        output_dir=out_dir / "tokenized_train",
        tokenizer_path=tokenizer_path,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        use_existing_tokenizer=reuse_train_tokenizer,
    )
    if not reuse_train_tokenizer:
        write_tokenizer_cache(
            tokenizer_path,
            train_sha256=raw_split.train_sha256,
            vocab_size=args.vocab_size,
            specials=specials,
        )
    expected_record_length = args.context_length + 1
    if train_corpus.sequence_length != expected_record_length:
        raise ValueError(f"unexpected train tokenized sequence length {train_corpus.sequence_length}")

    val_corpus = None
    if raw_split.val_blocks > 0:
        val_corpus = prepare_tokenized_corpus(
            split_name="val",
            input_path=raw_split.val_path,
            output_dir=out_dir / "tokenized_val",
            tokenizer_path=tokenizer_path,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            use_existing_tokenizer=True,
        )
        if val_corpus.sequence_length != expected_record_length:
            raise ValueError(f"unexpected val tokenized sequence length {val_corpus.sequence_length}")

    paths = write_gdds_dataset(
        out_dir=out_dir,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        max_shard_bytes=args.max_shard_bytes,
    )
    write_manifest(
        out_dir=out_dir,
        source=args.source,
        tokenizer_path=tokenizer_path,
        raw_split=raw_split,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        gdds_paths=paths,
        max_shard_bytes=args.max_shard_bytes,
        vocab_size=args.vocab_size,
        split_seed=args.split_seed,
        split_block_chars=args.split_block_chars,
        val_samples_target=args.val_samples,
        val_fraction=args.val_fraction,
    )

    total_samples = sum(
        read_gdds_header(path).samples for split_paths in paths.values() for path in split_paths
    )
    train_samples = sum(read_gdds_header(path).samples for path in paths["train"])
    val_samples = sum(read_gdds_header(path).samples for path in paths.get("val", []))
    print(f"Wrote {total_samples} GPT LM samples to {out_dir} (train={train_samples}, val={val_samples})")
    for split_paths in paths.values():
        for path in split_paths:
            print(path)
    print(out_dir / "manifest.json")

    if args.preview_samples > 0:
        show_random_samples(
            shards=paths["train"],
            tokenizer_path=tokenizer_path,
            total_samples=train_samples,
            count=args.preview_samples,
            seed=args.preview_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
