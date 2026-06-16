#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Token-level corpus/bigram diagnostics for examples/gpt_lm.

Reads the actual GDDS rows used by training/eval, including target masks, so the
reported token entropies are directly comparable to GPT LM cross-entropy losses
(nats/token).  Also converts losses to bits/byte using tokenizer byte lengths.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

GDDS_MAGIC = b"GDDSv1\0\0"
GDDS_HEADER = struct.Struct("<8sIIIIQQQQQQ")
GDDS_FIELD_DESC = struct.Struct("<64sII8qQQQ")
GDDS_INDEX = struct.Struct("<QQ")
GDDS_RECORD_HEADER = struct.Struct("<4sHHIQ")
GDDS_RECORD_FIELD = struct.Struct("<HHI8qQQ")
DTYPE_CODES = {"f16": 1, "bf16": 2, "f32": 3, "i32": 4, "u8": 5}
DTYPE_NAMES = {v: k for k, v in DTYPE_CODES.items()}
DTYPE_NP = {1: np.dtype("<f2"), 2: np.dtype("<u2"), 3: np.dtype("<f4"), 4: np.dtype("<i4"), 5: np.dtype("u1")}
DTYPE_SIZE = {1: 2, 2: 2, 3: 4, 4: 4, 5: 1}


@dataclass(frozen=True)
class Field:
    name: str
    dtype: int
    rank: int
    shape: tuple[int, ...]
    flags: int


@dataclass(frozen=True)
class Shard:
    path: Path
    fields: list[Field]
    n_samples: int
    index_offset: int
    data_offset: int
    data_nbytes: int
    schema_hash: int


def read_shard_header(path: Path) -> Shard:
    with path.open("rb") as f:
        raw = f.read(GDDS_HEADER.size)
        if len(raw) != GDDS_HEADER.size:
            raise ValueError(f"short GDDS header: {path}")
        magic, version, header_size, field_count, _reserved, n_samples, schema_offset, index_offset, data_offset, data_nbytes, schema_hash = GDDS_HEADER.unpack(raw)
        if magic != GDDS_MAGIC or version != 1 or header_size != 128:
            raise ValueError(f"bad GDDS header: {path}")
        f.seek(schema_offset)
        fields: list[Field] = []
        for _ in range(field_count):
            raw = f.read(GDDS_FIELD_DESC.size)
            name_raw, dtype, rank, *rest = GDDS_FIELD_DESC.unpack(raw)
            dims = tuple(int(x) for x in rest[:8])
            flags = int(rest[8])
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            fields.append(Field(name, dtype, rank, dims[:rank], flags))
    return Shard(path, fields, int(n_samples), int(index_offset), int(data_offset), int(data_nbytes), int(schema_hash))


def read_index(f: BinaryIO, shard: Shard, sample_idx: int) -> tuple[int, int]:
    f.seek(shard.index_offset + sample_idx * GDDS_INDEX.size)
    raw = f.read(GDDS_INDEX.size)
    if len(raw) != GDDS_INDEX.size:
        raise ValueError(f"short GDDS index {shard.path} sample {sample_idx}")
    off, nbytes = GDDS_INDEX.unpack(raw)
    return int(off), int(nbytes)


def decode_record(shard: Shard, raw: bytes) -> dict[str, np.ndarray]:
    if len(raw) < GDDS_RECORD_HEADER.size:
        raise ValueError("short GDDS record")
    magic, n_fields, _reserved, header_nbytes, payload_nbytes = GDDS_RECORD_HEADER.unpack_from(raw, 0)
    if magic != b"GDDR":
        raise ValueError("bad GDDS record magic")
    payload_start = int(header_nbytes)
    payload_end = payload_start + int(payload_nbytes)
    if payload_end > len(raw):
        raise ValueError("short GDDS record payload")
    out: dict[str, np.ndarray] = {}
    pos = GDDS_RECORD_HEADER.size
    for _ in range(n_fields):
        field_id, rank, _flags, *rest = GDDS_RECORD_FIELD.unpack_from(raw, pos)
        pos += GDDS_RECORD_FIELD.size
        dims = tuple(int(x) for x in rest[:8])[:rank]
        off = int(rest[8])
        nbytes = int(rest[9])
        field = shard.fields[field_id]
        dtype = DTYPE_NP[field.dtype]
        expected = int(np.prod(dims, dtype=np.int64)) * DTYPE_SIZE[field.dtype]
        if expected != nbytes:
            raise ValueError(f"field byte mismatch {field.name}: expected={expected} got={nbytes}")
        data = memoryview(raw)[payload_start + off : payload_start + off + nbytes]
        out[field.name] = np.frombuffer(data, dtype=dtype).reshape(dims)
    return out


def iter_split_samples(data_dir: Path, split: str, *, limit: int | None = None) -> Iterator[dict[str, np.ndarray]]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    split_info = manifest["splits"][split]
    yielded = 0
    for shard_name in split_info["shards"]:
        shard = read_shard_header(data_dir / shard_name)
        with shard.path.open("rb") as f:
            for i in range(shard.n_samples):
                if limit is not None and yielded >= limit:
                    return
                off, nbytes = read_index(f, shard, i)
                f.seek(off)
                raw = f.read(nbytes)
                if len(raw) != nbytes:
                    raise ValueError(f"short GDDS record data {shard.path} sample {i}")
                yielded += 1
                yield decode_record(shard, raw)


def tokenizer_byte_lengths(tokenizer_path: Path) -> tuple[np.ndarray, list[str], set[int]]:
    tok = json.loads(tokenizer_path.read_text())
    vocab_size = int(tok["vocab_size"])
    lengths = np.zeros(vocab_size, dtype=np.int64)
    labels = [""] * vocab_size
    special: set[int] = set()
    for item in tok["tokens"]:
        i = int(item["id"])
        kind = item["kind"]
        if kind == "special":
            labels[i] = item["text"]
            special.add(i)
            lengths[i] = 0
        else:
            hx = item.get("hex", "")
            lengths[i] = len(hx) // 2
            try:
                piece = bytes.fromhex(hx).decode("utf-8")
                labels[i] = piece.replace("\n", "\\n")
            except Exception:
                labels[i] = f"0x{hx}"
    return lengths, labels, special


@dataclass
class SplitCounts:
    name: str
    samples: int
    tokens: int
    valid: int
    masked: int
    target_bytes: int
    target_special: int
    unigram: np.ndarray
    context: np.ndarray
    bigram: np.ndarray
    bucket_valid: dict[str, int]
    bucket_bytes: dict[str, int]


def bucket_name(pos: np.ndarray) -> np.ndarray:
    # output codes: 0 [0,15], 1 [16,31], 2 [32,63], 3 [64,127], 4 [128,255], 5 [256,511]
    bins = np.zeros(pos.shape, dtype=np.int8)
    bins[pos >= 16] = 1
    bins[pos >= 32] = 2
    bins[pos >= 64] = 3
    bins[pos >= 128] = 4
    bins[pos >= 256] = 5
    return bins

BUCKET_LABELS = ["000-015", "016-031", "032-063", "064-127", "128-255", "256-511"]


def count_split(data_dir: Path, split: str, vocab_size: int, token_bytes: np.ndarray, *, limit: int | None = None, chunk_items: int = 2_000_000) -> SplitCounts:
    unigram = np.zeros(vocab_size, dtype=np.int64)
    context = np.zeros(vocab_size, dtype=np.int64)
    bigram = np.zeros(vocab_size * vocab_size, dtype=np.int64)
    pending: list[np.ndarray] = []
    pending_n = 0
    samples = tokens = valid = masked = target_bytes = target_special = 0
    bucket_valid = {label: 0 for label in BUCKET_LABELS}
    bucket_bytes = {label: 0 for label in BUCKET_LABELS}

    def flush() -> None:
        nonlocal pending_n
        if not pending:
            return
        flat = np.concatenate(pending)
        bigram[:] += np.bincount(flat, minlength=vocab_size * vocab_size).astype(np.int64, copy=False)
        pending.clear()
        pending_n = 0

    for rec in iter_split_samples(data_dir, split, limit=limit):
        inp = rec["input_ids"].astype(np.int64, copy=False)
        tgt = rec["target_ids"].astype(np.int64, copy=False)
        pos = rec.get("positions")
        mask = tgt >= 0
        x = inp[mask]
        y = tgt[mask]
        samples += 1
        tokens += int(inp.size)
        n_valid = int(y.size)
        valid += n_valid
        masked += int(tgt.size - n_valid)
        if n_valid:
            target_bytes += int(token_bytes[y].sum())
            target_special += int((token_bytes[y] == 0).sum())
            unigram += np.bincount(y, minlength=vocab_size).astype(np.int64, copy=False)
            context += np.bincount(x, minlength=vocab_size).astype(np.int64, copy=False)
            flat = x * vocab_size + y
            pending.append(flat.astype(np.int64, copy=False))
            pending_n += n_valid
            if pos is not None:
                codes = bucket_name(pos[mask].astype(np.int64, copy=False))
                for code, label in enumerate(BUCKET_LABELS):
                    m = codes == code
                    if m.any():
                        bucket_valid[label] += int(m.sum())
                        bucket_bytes[label] += int(token_bytes[y[m]].sum())
            if pending_n >= chunk_items:
                flush()
    flush()
    return SplitCounts(split, samples, tokens, valid, masked, target_bytes, target_special, unigram, context, bigram.reshape((vocab_size, vocab_size)), bucket_valid, bucket_bytes)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    nz = counts[counts > 0].astype(np.float64)
    p = nz / float(total)
    return float(-(p * np.log(p)).sum())


def empirical_bigram_entropy(bigram: np.ndarray, context: np.ndarray) -> float:
    total = int(bigram.sum())
    if total == 0:
        return float("nan")
    rows, cols = np.nonzero(bigram)
    c = bigram[rows, cols].astype(np.float64)
    denom = context[rows].astype(np.float64)
    return float(-(c * (np.log(c) - np.log(denom))).sum() / float(total))


def eval_unigram_nll(eval_counts: np.ndarray, train_counts: np.ndarray, alpha: float) -> float:
    v = train_counts.size
    denom = float(train_counts.sum()) + alpha * v
    logp = np.log(train_counts.astype(np.float64) + alpha) - math.log(denom)
    total = int(eval_counts.sum())
    return float(-(eval_counts.astype(np.float64) * logp).sum() / total)


def eval_bigram_add_alpha(eval_bigram: np.ndarray, train_bigram: np.ndarray, train_context: np.ndarray, alpha: float) -> float:
    v = train_context.size
    rows, cols = np.nonzero(eval_bigram)
    c_eval = eval_bigram[rows, cols].astype(np.float64)
    c_train = train_bigram[rows, cols].astype(np.float64)
    denom = train_context[rows].astype(np.float64) + alpha * v
    logp = np.log(c_train + alpha) - np.log(denom)
    return float(-(c_eval * logp).sum() / float(eval_bigram.sum()))


def eval_bigram_interp(eval_bigram: np.ndarray, train_bigram: np.ndarray, train_context: np.ndarray, train_unigram: np.ndarray, k: float, alpha: float) -> float:
    v = train_context.size
    p_uni = (train_unigram.astype(np.float64) + alpha) / (float(train_unigram.sum()) + alpha * v)
    rows, cols = np.nonzero(eval_bigram)
    c_eval = eval_bigram[rows, cols].astype(np.float64)
    ctx = train_context[rows].astype(np.float64)
    c_train = train_bigram[rows, cols].astype(np.float64)
    mle = np.zeros_like(c_train, dtype=np.float64)
    seen_ctx = ctx > 0
    mle[seen_ctx] = c_train[seen_ctx] / ctx[seen_ctx]
    lam = ctx / (ctx + k)
    p = lam * mle + (1.0 - lam) * p_uni[cols]
    # p should be positive due to unigram smoothing; guard anyway.
    p = np.maximum(p, 1e-300)
    return float(-(c_eval * np.log(p)).sum() / float(eval_bigram.sum()))


def fmt_loss(nats: float, byte_per_token: float) -> str:
    bits_tok = nats / math.log(2.0)
    bpb = bits_tok / byte_per_token if byte_per_token > 0 else float("nan")
    ppl = math.exp(nats) if nats < 50 else float("inf")
    return f"{nats:.6f} nats/tok  {bits_tok:.4f} bits/tok  {bpb:.4f} bits/byte  ppl={ppl:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="examples/gpt_lm/data")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--model-train-loss", type=float, default=1.65)
    ap.add_argument("--model-val-loss", type=float, default=1.88)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--interp-k", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else data_dir / "tokenizer-v2048.json"
    token_bytes, labels, special = tokenizer_byte_lengths(tokenizer_path)
    if token_bytes.size != args.vocab_size:
        raise SystemExit(f"tokenizer vocab size {token_bytes.size} != --vocab-size {args.vocab_size}")

    train = count_split(data_dir, "train", args.vocab_size, token_bytes, limit=args.limit_train)
    val = count_split(data_dir, "val", args.vocab_size, token_bytes, limit=args.limit_val)

    train_bpt = train.target_bytes / train.valid
    val_bpt = val.target_bytes / val.valid
    train_uni_emp = entropy_from_counts(train.unigram)
    val_uni_emp = entropy_from_counts(val.unigram)
    train_bi_emp = empirical_bigram_entropy(train.bigram, train.context)
    val_bi_emp = empirical_bigram_entropy(val.bigram, val.context)
    uni_train = eval_unigram_nll(train.unigram, train.unigram, args.alpha)
    uni_val = eval_unigram_nll(val.unigram, train.unigram, args.alpha)
    bi_add_train = eval_bigram_add_alpha(train.bigram, train.bigram, train.context, args.alpha)
    bi_add_val = eval_bigram_add_alpha(val.bigram, train.bigram, train.context, args.alpha)
    bi_int_train = eval_bigram_interp(train.bigram, train.bigram, train.context, train.unigram, args.interp_k, args.alpha)
    bi_int_val = eval_bigram_interp(val.bigram, train.bigram, train.context, train.unigram, args.interp_k, args.alpha)

    result = {
        "data_dir": str(data_dir),
        "tokenizer": str(tokenizer_path),
        "splits": {
            "train": {
                "samples": train.samples,
                "tokens": train.tokens,
                "valid_targets": train.valid,
                "masked_targets": train.masked,
                "target_bytes": train.target_bytes,
                "target_special": train.target_special,
                "bytes_per_valid_target": train_bpt,
                "bucket_valid": train.bucket_valid,
                "bucket_bytes": train.bucket_bytes,
            },
            "val": {
                "samples": val.samples,
                "tokens": val.tokens,
                "valid_targets": val.valid,
                "masked_targets": val.masked,
                "target_bytes": val.target_bytes,
                "target_special": val.target_special,
                "bytes_per_valid_target": val_bpt,
                "bucket_valid": val.bucket_valid,
                "bucket_bytes": val.bucket_bytes,
            },
        },
        "losses_nats_per_token": {
            "train_unigram_empirical": train_uni_emp,
            "val_unigram_empirical": val_uni_emp,
            "train_bigram_empirical": train_bi_emp,
            "val_bigram_empirical_oracle": val_bi_emp,
            "train_unigram_add_alpha": uni_train,
            "val_unigram_add_alpha": uni_val,
            "train_bigram_add_alpha": bi_add_train,
            "val_bigram_add_alpha": bi_add_val,
            "train_bigram_interpolated": bi_int_train,
            "val_bigram_interpolated": bi_int_val,
            "model_train_loss_input": args.model_train_loss,
            "model_val_loss_input": args.model_val_loss,
        },
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"data_dir: {data_dir}")
    print(f"tokenizer: {tokenizer_path} vocab={args.vocab_size} specials={sorted(special)}")
    for s, bpt in [(train, train_bpt), (val, val_bpt)]:
        print(
            f"{s.name}: samples={s.samples:,} row_tokens={s.tokens:,} valid_targets={s.valid:,} "
            f"masked={s.masked:,} target_bytes={s.target_bytes:,} bytes/valid_target={bpt:.4f} "
            f"special_targets={s.target_special:,}"
        )
        print("  position buckets (input position predicting target):")
        for label in BUCKET_LABELS:
            n = s.bucket_valid[label]
            bp = s.bucket_bytes[label] / n if n else float("nan")
            print(f"    {label}: valid={n:,} bytes={s.bucket_bytes[label]:,} bytes/tok={bp:.4f}")
    print("\nEntropy / NLL (comparable to model loss; lower is better):")
    print(f"  train unigram empirical:       {fmt_loss(train_uni_emp, train_bpt)}")
    print(f"  val   unigram empirical oracle:{fmt_loss(val_uni_emp, val_bpt)}")
    print(f"  train bigram empirical:        {fmt_loss(train_bi_emp, train_bpt)}")
    print(f"  val   bigram empirical oracle: {fmt_loss(val_bi_emp, val_bpt)}")
    print(f"  train unigram add-{args.alpha:g}:        {fmt_loss(uni_train, train_bpt)}")
    print(f"  val   unigram add-{args.alpha:g}:        {fmt_loss(uni_val, val_bpt)}")
    print(f"  train bigram add-{args.alpha:g}:         {fmt_loss(bi_add_train, train_bpt)}")
    print(f"  val   bigram add-{args.alpha:g}:         {fmt_loss(bi_add_val, val_bpt)}")
    print(f"  train bigram interp k={args.interp_k:g}: {fmt_loss(bi_int_train, train_bpt)}")
    print(f"  val   bigram interp k={args.interp_k:g}: {fmt_loss(bi_int_val, val_bpt)}")
    print("\nModel losses converted with same bytes/valid-target:")
    print(f"  model train loss={args.model_train_loss:g}: {fmt_loss(args.model_train_loss, train_bpt)}")
    print(f"  model val   loss={args.model_val_loss:g}: {fmt_loss(args.model_val_loss, val_bpt)}")

    # A few most common token targets help sanity-check tokenizer/data.
    top = np.argsort(train.unigram)[-20:][::-1]
    print("\nTop train target tokens:")
    for i in top:
        print(f"  id={int(i):4d} count={int(train.unigram[i]):9d} bytes={int(token_bytes[i])} piece={labels[i]!r}")


if __name__ == "__main__":
    main()
