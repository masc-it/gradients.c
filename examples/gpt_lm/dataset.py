#!/usr/bin/env python3
"""Prepare a tiny packed GPT language-modeling GDDS dataset.

The default source is ``~/projects/dnn.c/docs/promessi_sposi.txt``.  The script:

1. trains a byte-level BPE tokenizer with a 2048 token vocabulary, unless a
   compatible tokenizer already exists;
2. uses the optimized C tokenizer to produce packed token records of
   ``context_length + 1`` tokens with one-token overlap for next-token
   prediction;
3. writes a GDDS split under ``examples/gpt_lm/data`` with shifted
   ``input_ids``/``target_ids`` tensors of length ``context_length``; and
4. prints decoded random samples loaded back from the produced GDDS shard.
"""

from __future__ import annotations

import argparse
import json
import random
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
DATA_FORMAT_VERSION = "gpt-lm-promessi-v1"


@dataclass(frozen=True)
class ShardHeader:
    samples: int
    index_offset: int
    schema_hash: int


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


def prepare_tokenized_corpus(
    *,
    source: Path,
    data_dir: Path,
    tokenizer_path: Path,
    context_length: int,
    vocab_size: int,
    retrain_tokenizer: bool,
) -> TokenizedCorpus:
    tokenized_dir = data_dir / "tokenized"
    ensure_clean_dir(tokenized_dir)
    use_existing = tokenizer_path.exists() and not retrain_tokenizer and tokenizer_compatible(
        tokenizer_path,
        vocab_size=vocab_size,
        specials=(IM_START, IM_END),
    )
    summary = tokenize_file_packed(
        tokenizer=tokenizer_path,
        input_path=source,
        output_dir=tokenized_dir,
        im_start=IM_START,
        im_end=IM_END,
        num_tokens_per_sequence=context_length + 1,
        use_tokenizer=use_existing,
        vocab_size=vocab_size,
        min_frequency=2,
        seed=17,
    )
    action = "reused" if use_existing else "trained"
    print(
        f"Tokenizer {action}: {summary['tokenizer']} "
        f"(records={summary['sequences']}, record_tokens={context_length + 1}, "
        f"tokens={summary['tokens']})"
    )
    return TokenizedCorpus.open(tokenized_dir)


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
    corpus: TokenizedCorpus,
    max_shard_bytes: int,
) -> list[Path]:
    if corpus.sequence_length is None or corpus.sequence_length < 3:
        raise ValueError("packed tokenized corpus must have sequence_length >= 3")
    context_length = corpus.sequence_length - 1
    fields = fields_for_context(context_length)
    remove_split_shards(out_dir, "train")
    writer = GddsSplitWriter(out_dir, "train", fields, max_shard_bytes=max_shard_bytes)
    try:
        for sample in iter_lm_samples(corpus):
            writer.write_sample(sample)
        paths = writer.finish()
    except BaseException:
        writer.abort()
        raise
    if not paths:
        raise ValueError("tokenizer produced no packed records")
    return paths


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
    tokenized_corpus: TokenizedCorpus,
    gdds_paths: Sequence[Path],
    max_shard_bytes: int,
    vocab_size: int,
) -> None:
    assert tokenized_corpus.sequence_length is not None
    context_length = tokenized_corpus.sequence_length - 1
    fields = fields_for_context(context_length)
    split_samples = sum(read_gdds_header(path).samples for path in gdds_paths)
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
            "hash": tokenized_corpus.manifest["tokenizer_hash"],
            "vocab_size": vocab_size,
            "im_start_id": tokenized_corpus.im_start_id,
            "im_end_id": tokenized_corpus.im_end_id,
            "digits": "always split before BPE merges",
        },
        "packing": {
            "record_length": tokenized_corpus.sequence_length,
            "context_length": context_length,
            "stride": tokenized_corpus.stride,
            "source_records": tokenized_corpus.num_sequences,
            "source_tokens": tokenized_corpus.num_tokens,
        },
        "splits": {
            "train": {
                "samples": split_samples,
                "shards": [path.name for path in gdds_paths],
            }
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

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / f"tokenizer-v{args.vocab_size}.json"

    corpus = prepare_tokenized_corpus(
        source=args.source,
        data_dir=out_dir,
        tokenizer_path=tokenizer_path,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        retrain_tokenizer=args.retrain_tokenizer,
    )
    expected_record_length = args.context_length + 1
    if corpus.sequence_length != expected_record_length:
        raise ValueError(f"unexpected tokenized sequence length {corpus.sequence_length}")

    paths = write_gdds_dataset(out_dir=out_dir, corpus=corpus, max_shard_bytes=args.max_shard_bytes)
    write_manifest(
        out_dir=out_dir,
        source=args.source,
        tokenizer_path=tokenizer_path,
        tokenized_corpus=corpus,
        gdds_paths=paths,
        max_shard_bytes=args.max_shard_bytes,
        vocab_size=args.vocab_size,
    )

    total_samples = sum(read_gdds_header(path).samples for path in paths)
    print(f"Wrote {total_samples} GPT LM samples to {out_dir}")
    for path in paths:
        print(path)
    print(out_dir / "manifest.json")

    if args.preview_samples > 0:
        show_random_samples(
            shards=paths,
            tokenizer_path=tokenizer_path,
            total_samples=total_samples,
            count=args.preview_samples,
            seed=args.preview_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
