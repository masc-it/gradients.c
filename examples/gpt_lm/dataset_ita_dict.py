#!/usr/bin/env python3
"""Prepare a packed GPT LM dataset from Italian dictionary definition JSON files.

The source directories contain one JSON file per generated dictionary entry.  The
``definizioni-clean`` files contain terms and definition strings.  The
``definizioni-clean-enriched`` files contain the same subset of terms with
examples attached to each definition.  This builder merges both directories,
prefers enriched entries when present, splits by *term* before tokenizer
training, trains the byte-level BPE tokenizer on the training text only, then
writes train/val GDDS shards compatible with ``examples/gpt_lm``.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
DEFAULT_CLEAN_DIR = Path("/Users/mascit/projects/DataFarmer/data/definizioni-clean")
DEFAULT_ENRICHED_DIR = Path("/Users/mascit/projects/DataFarmer/data/definizioni-clean-enriched")
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_MAX_EXAMPLES_PER_DEFINITION = 3
MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
DATA_FORMAT_VERSION = "gpt-lm-ita-dict-v1"


@dataclass(frozen=True)
class ShardHeader:
    samples: int
    index_offset: int
    schema_hash: int


@dataclass
class DefinitionEntry:
    definition: str
    examples: list[str] = field(default_factory=list)


@dataclass
class TermEntry:
    term: str
    definitions: list[DefinitionEntry] = field(default_factory=list)
    source_ids: set[str] = field(default_factory=set)
    enriched: bool = False


@dataclass(frozen=True)
class CorpusSplit:
    train_terms: list[str]
    val_terms: list[str]
    train_chars: int
    val_chars: int
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text: str) -> str:
    return normalize_space(text).casefold()


def iter_json_files(directory: Path) -> Iterable[Path]:
    yield from sorted(path for path in directory.glob("*.json") if path.name != "status.json")


def parse_clean_file(path: Path) -> TermEntry | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipping invalid JSON {path}: {exc}", file=sys.stderr)
        return None
    term = normalize_space(data.get("term"))
    raw_definitions = data.get("definitions")
    if not term or not isinstance(raw_definitions, list):
        print(f"warning: skipping malformed clean entry {path}", file=sys.stderr)
        return None
    definitions = [DefinitionEntry(definition=d) for d in map(normalize_space, raw_definitions) if d]
    if not definitions:
        return None
    return TermEntry(term=term, definitions=definitions, source_ids={path.stem}, enriched=False)


def parse_enriched_file(path: Path, max_examples_per_definition: int) -> TermEntry | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipping invalid JSON {path}: {exc}", file=sys.stderr)
        return None
    term = normalize_space(data.get("term"))
    raw_definitions = data.get("definitions")
    if not term or not isinstance(raw_definitions, list):
        print(f"warning: skipping malformed enriched entry {path}", file=sys.stderr)
        return None
    definitions: list[DefinitionEntry] = []
    for item in raw_definitions:
        if isinstance(item, str):
            definition = normalize_space(item)
            examples: list[str] = []
        elif isinstance(item, Mapping):
            definition = normalize_space(item.get("definition"))
            raw_examples = item.get("examples")
            examples = []
            if isinstance(raw_examples, list):
                for example in raw_examples:
                    normalized = normalize_space(example)
                    if normalized and normalize_key(normalized) != normalize_key(definition):
                        examples.append(normalized)
        else:
            continue
        if not definition:
            continue
        definitions.append(
            DefinitionEntry(
                definition=definition,
                examples=dedupe_texts(examples, max_items=max_examples_per_definition),
            )
        )
    if not definitions:
        return None
    return TermEntry(term=term, definitions=definitions, source_ids={path.stem}, enriched=True)


def dedupe_texts(texts: Iterable[str], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        text = normalize_space(text)
        if not text:
            continue
        key = normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if max_items is not None and max_items >= 0 and len(out) >= max_items:
            break
    return out


def merge_term_entry(target: TermEntry, source: TermEntry, max_examples_per_definition: int) -> None:
    target.source_ids.update(source.source_ids)
    target.enriched = target.enriched or source.enriched
    by_definition = {normalize_key(defn.definition): defn for defn in target.definitions}
    for defn in source.definitions:
        key = normalize_key(defn.definition)
        if key not in by_definition:
            copied = DefinitionEntry(
                definition=defn.definition,
                examples=dedupe_texts(defn.examples, max_items=max_examples_per_definition),
            )
            target.definitions.append(copied)
            by_definition[key] = copied
            continue
        existing = by_definition[key]
        existing.examples = dedupe_texts(
            [*existing.examples, *defn.examples],
            max_items=max_examples_per_definition,
        )


def load_entries(clean_dir: Path, enriched_dir: Path, max_examples_per_definition: int) -> tuple[list[TermEntry], dict[str, int]]:
    clean_by_id: dict[str, TermEntry] = {}
    enriched_by_id: dict[str, TermEntry] = {}
    for path in iter_json_files(clean_dir):
        entry = parse_clean_file(path)
        if entry is not None:
            clean_by_id[path.stem] = entry
    for path in iter_json_files(enriched_dir):
        entry = parse_enriched_file(path, max_examples_per_definition=max_examples_per_definition)
        if entry is not None:
            enriched_by_id[path.stem] = entry

    merged_by_term: dict[str, TermEntry] = {}
    all_ids = sorted(set(clean_by_id) | set(enriched_by_id))
    for source_id in all_ids:
        # Enriched files are generated from clean files and carry examples, so
        # use them as the canonical entry whenever present.  Clean-only ids are
        # still included with definitions and no examples.
        source = enriched_by_id.get(source_id) or clean_by_id.get(source_id)
        if source is None:
            continue
        term_key = normalize_key(source.term)
        if term_key not in merged_by_term:
            merged_by_term[term_key] = TermEntry(
                term=source.term,
                definitions=[],
                source_ids=set(),
                enriched=False,
            )
        merge_term_entry(merged_by_term[term_key], source, max_examples_per_definition)

    entries = sorted(merged_by_term.values(), key=lambda e: normalize_key(e.term))
    stats = {
        "clean_files": len(clean_by_id),
        "enriched_files": len(enriched_by_id),
        "overlap_files": len(set(clean_by_id) & set(enriched_by_id)),
        "clean_only_files": len(set(clean_by_id) - set(enriched_by_id)),
        "terms": len(entries),
        "definitions": sum(len(e.definitions) for e in entries),
        "examples": sum(len(d.examples) for e in entries for d in e.definitions),
        "enriched_terms": sum(1 for e in entries if e.enriched),
    }
    return entries, stats


def format_entry(entry: TermEntry) -> str:
    lines = ["### Voce", f"Termine: {entry.term}", "Definizioni:"]
    for i, definition in enumerate(entry.definitions, start=1):
        lines.append(f"{i}. {definition.definition}")
        if definition.examples:
            lines.append("   Esempi:")
            for example in definition.examples:
                lines.append(f"   - {example}")
    lines.append("### Fine voce")
    return "\n".join(lines)


def entry_chars(entry: TermEntry) -> int:
    return len(format_entry(entry))


def split_entries(entries: Sequence[TermEntry], val_fraction: float, split_seed: int) -> tuple[list[TermEntry], list[TermEntry]]:
    if not entries:
        raise ValueError("no entries to split")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    if len(entries) == 1 or val_fraction == 0.0:
        return list(entries), []

    total_chars = sum(entry_chars(entry) for entry in entries)
    target_val_chars = max(1, int(round(total_chars * val_fraction)))
    shuffled = list(entries)
    random.Random(split_seed).shuffle(shuffled)
    val_keys: set[str] = set()
    val_chars = 0
    for entry in shuffled:
        if len(val_keys) >= len(entries) - 1:
            break
        key = normalize_key(entry.term)
        val_keys.add(key)
        val_chars += entry_chars(entry)
        if val_chars >= target_val_chars:
            break
    train = [entry for entry in entries if normalize_key(entry.term) not in val_keys]
    val = [entry for entry in entries if normalize_key(entry.term) in val_keys]
    if not train:
        raise ValueError("split produced empty training set")
    return train, val


def write_text_split(out_dir: Path, train_entries: Sequence[TermEntry], val_entries: Sequence[TermEntry]) -> CorpusSplit:
    raw_dir = out_dir / "raw_ita_dict"
    ensure_clean_dir(raw_dir)
    train_path = raw_dir / "train.txt"
    val_path = raw_dir / "val.txt"
    train_text = format_entries(train_entries)
    val_text = format_entries(val_entries)
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    return CorpusSplit(
        train_terms=[entry.term for entry in train_entries],
        val_terms=[entry.term for entry in val_entries],
        train_chars=len(train_text),
        val_chars=len(val_text),
        train_sha256=file_sha256(train_path),
        val_sha256=file_sha256(val_path) if val_text else "",
    )


def format_entries(entries: Sequence[TermEntry]) -> str:
    text = "\n\n".join(format_entry(entry) for entry in entries)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def prepare_tokenized_corpus(
    *,
    split_name: str,
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    context_length: int,
    vocab_size: int,
    min_frequency: int,
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
        min_frequency=min_frequency,
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
    clean_dir: Path,
    enriched_dir: Path,
    tokenizer_path: Path,
    split: CorpusSplit,
    source_stats: Mapping[str, int],
    train_corpus: TokenizedCorpus,
    val_corpus: TokenizedCorpus | None,
    gdds_paths: Mapping[str, Sequence[Path]],
    max_shard_bytes: int,
    vocab_size: int,
    context_length: int,
    val_fraction: float,
    split_seed: int,
    max_examples_per_definition: int,
) -> None:
    assert train_corpus.sequence_length is not None
    fields = fields_for_context(context_length)
    split_samples = {
        split_name: sum(read_gdds_header(path).samples for path in paths)
        for split_name, paths in gdds_paths.items()
    }
    val_records = 0 if val_corpus is None else val_corpus.num_sequences
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    manifest = {
        "format": "GDDS",
        "version": 1,
        "schema_hash": f"0x{schema_hash(fields):016x}",
        "fields": [field_metadata(field_spec) for field_spec in fields],
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
            "trained_on": "raw_ita_dict/train.txt",
            "train_sha256": split.train_sha256,
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
            split_name: {
                "samples": split_samples.get(split_name, 0),
                "shards": [path.name for path in paths],
            }
            for split_name, paths in gdds_paths.items()
        },
        "raw_split": {
            "mode": "term_random_before_tokenizer_training",
            "val_fraction": val_fraction,
            "split_seed": split_seed,
            "train_terms": len(split.train_terms),
            "val_terms": len(split.val_terms),
            "train_chars": split.train_chars,
            "val_chars": split.val_chars,
            "train_sha256": split.train_sha256,
            "val_sha256": split.val_sha256,
            "train_path": "raw_ita_dict/train.txt",
            "val_path": "raw_ita_dict/val.txt",
        },
        "dictionary_format": {
            "entry_separator": "blank line",
            "entry_header": "### Voce",
            "term_line": "Termine: <lemma>",
            "definitions_header": "Definizioni:",
            "definition_line": "<n>. <definition>",
            "examples_header": "   Esempi:",
            "example_line": "   - <example>",
            "entry_footer": "### Fine voce",
            "max_examples_per_definition": max_examples_per_definition,
        },
        "source": {
            "clean_dir": str(clean_dir),
            "enriched_dir": str(enriched_dir),
            **dict(source_stats),
        },
        "prep": {
            "format_version": DATA_FORMAT_VERSION,
            "max_shard_bytes": max_shard_bytes,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "split_terms.json").write_text(
        json.dumps({"train": split.train_terms, "val": split.val_terms}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def show_random_samples(shards: Sequence[Path], tokenizer_path: Path, total_samples: int, count: int, seed: int) -> None:
    if count <= 0 or total_samples <= 0:
        return
    decoder = TokenDecoder(tokenizer_path)
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_samples), k=min(count, total_samples)))
    print(f"\nDecoded random train samples ({len(selected)}):")
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
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--enriched-dir", type=Path, default=DEFAULT_ENRICHED_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--max-examples-per-definition",
        type=int,
        default=DEFAULT_MAX_EXAMPLES_PER_DEFINITION,
        help="use -1 to keep all examples; default caps examples so they do not swamp definitions",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--preview-seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(args.clean_dir)
    if not args.enriched_dir.is_dir():
        raise FileNotFoundError(args.enriched_dir)
    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if args.vocab_size < 258:
        raise ValueError("--vocab-size must be at least 258 for byte tokens plus two specials")
    if args.min_frequency <= 0:
        raise ValueError("--min-frequency must be positive")
    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1")
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")

    entries, stats = load_entries(
        clean_dir=args.clean_dir,
        enriched_dir=args.enriched_dir,
        max_examples_per_definition=args.max_examples_per_definition,
    )
    if not entries:
        raise ValueError("no dictionary entries loaded")
    print(
        "Loaded dictionary: "
        f"terms={stats['terms']} definitions={stats['definitions']} examples={stats['examples']} "
        f"clean_files={stats['clean_files']} enriched_files={stats['enriched_files']} "
        f"clean_only_files={stats['clean_only_files']}"
    )

    train_entries, val_entries = split_entries(entries, args.val_fraction, args.split_seed)
    ensure_clean_dir(args.out_dir)
    tokenizer_path = args.out_dir / f"tokenizer-v{args.vocab_size}.json"
    split = write_text_split(args.out_dir, train_entries, val_entries)
    print(
        "Raw term split: "
        f"train_terms={len(train_entries)} val_terms={len(val_entries)} "
        f"train_chars={split.train_chars} val_chars={split.val_chars} "
        f"val_fraction={args.val_fraction:.3f} seed={args.split_seed}"
    )

    train_corpus = prepare_tokenized_corpus(
        split_name="train",
        input_path=args.out_dir / "raw_ita_dict" / "train.txt",
        output_dir=args.out_dir / "tokenized_train",
        tokenizer_path=tokenizer_path,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        use_existing_tokenizer=False,
    )
    val_corpus = None
    if val_entries:
        val_corpus = prepare_tokenized_corpus(
            split_name="val",
            input_path=args.out_dir / "raw_ita_dict" / "val.txt",
            output_dir=args.out_dir / "tokenized_val",
            tokenizer_path=tokenizer_path,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            use_existing_tokenizer=True,
        )

    expected_record_length = args.context_length + 1
    if train_corpus.sequence_length != expected_record_length:
        raise ValueError(f"unexpected train sequence length {train_corpus.sequence_length}")
    if val_corpus is not None and val_corpus.sequence_length != expected_record_length:
        raise ValueError(f"unexpected val sequence length {val_corpus.sequence_length}")

    paths = write_gdds_dataset(
        out_dir=args.out_dir,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        max_shard_bytes=args.max_shard_bytes,
    )
    write_manifest(
        out_dir=args.out_dir,
        clean_dir=args.clean_dir,
        enriched_dir=args.enriched_dir,
        tokenizer_path=tokenizer_path,
        split=split,
        source_stats=stats,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        gdds_paths=paths,
        max_shard_bytes=args.max_shard_bytes,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_examples_per_definition=args.max_examples_per_definition,
    )

    total_samples = sum(read_gdds_header(path).samples for split_paths in paths.values() for path in split_paths)
    train_samples = sum(read_gdds_header(path).samples for path in paths["train"])
    val_samples = sum(read_gdds_header(path).samples for path in paths.get("val", []))
    print(f"Wrote {total_samples} GPT LM samples to {args.out_dir} (train={train_samples}, val={val_samples})")
    for split_paths in paths.values():
        for path in split_paths:
            print(path)
    print(args.out_dir / "manifest.json")

    show_random_samples(
        paths["train"],
        tokenizer_path,
        train_samples,
        count=args.preview_samples,
        seed=args.preview_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
