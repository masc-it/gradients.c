#!/usr/bin/env python3
"""Prepare separate Italian dictionary mid-training and quiz/SFT GDDS datasets.

This builder is intentionally close to ``dataset_ita_dict_v2.py`` but it does
not train a tokenizer and it keeps the two data stages separate:

* ``midtrain`` uses base dictionary Markdown pages matching ``*.md`` plus
  ``*.story.md`` siblings.  It excludes ``*.quiz.md`` siblings.  Base pages
  preserve the source Markdown structure, except for the v2 normalization that
  removes the leading ``# `` before ``Termine:``; story pages are emitted as
  standalone ``## Story`` documents.
* ``sft`` uses only quiz Markdown pages matching ``*.quiz.md``.  Each quiz file
  is split on Markdown horizontal rules (``---``), producing one Q&A document
  per block.

Both outputs use an existing tokenizer supplied with ``--tokenizer``.  The
resulting datasets use the same compact GPT-LM GDDS storage as
``dataset_ita_dict_v2.py``: each sample stores ``uint16`` ``tokens`` of length
``context_length + 1`` plus ``segment_lengths`` metadata; the C dataloader
expands that into ``input_ids``/``target_ids``/``positions``/``cu_seqlens`` at
runtime.  The default packing context/sequence length for v3 is 2048 tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gdds_utils import field_metadata, schema_hash  # noqa: E402
from dataset_ita_dict_v2 import (  # noqa: E402
    DEFAULT_QUIZ_SUFFIX,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SOURCE_GLOB,
    DEFAULT_STORY_SUFFIX,
    DEFAULT_VAL_FRACTION,
    IM_END,
    IM_START,
    MAX_SHARD_BYTES,
    PAD,
    TERM_RE,
    TextDocument,
    clean_inline_markdown,
    ensure_clean_dir,
    fields_for_context,
    file_sha256,
    filename_slug_term,
    format_sibling_document,
    normalize_key,
    normalize_story_text,
    normalize_term_markdown,
    prepare_tokenized_corpus,
    read_gdds_header,
    show_random_samples,
    source_sort_key,
    split_quiz_items,
    storage_fields_for_context,
    terms_match_filename,
    tokenizer_special_id,
    write_gdds_dataset,
)
from tok_utils import TokenizedCorpus  # noqa: E402

DATA_FORMAT_VERSION = "gpt-lm-ita-dict-v3-md-story-and-quiz-existing-tokenizer-compact-u16"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data_v3"
DEFAULT_MIDTRAIN_SUBDIR = "midtrain"
DEFAULT_SFT_SUBDIR = "sft"
DEFAULT_QUIZ_GLOB = "*.quiz.md"
DEFAULT_STORY_GLOB = "*.story.md"
DEFAULT_CONTEXT_LENGTH = 2048


@dataclass(frozen=True)
class TokenizerInfo:
    path: Path
    vocab_size: int
    sha256: str
    spec_hash: str
    pad_id: int
    im_start_id: int
    im_end_id: int


@dataclass(frozen=True)
class DocumentSplit:
    train_terms: list[str]
    val_terms: list[str]
    train_chars: int
    val_chars: int
    train_documents: int
    val_documents: int
    train_sha256: str
    val_sha256: str
    train_jsonl_sha256: str
    val_jsonl_sha256: str


@dataclass(frozen=True)
class DatasetSummary:
    kind: str
    out_dir: Path
    train_samples: int
    val_samples: int
    train_documents: int
    val_documents: int
    train_tokens: int
    val_tokens: int


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_base_markdown_files(
    directory: Path,
    pattern: str,
    *,
    story_suffix: str,
    quiz_suffix: str,
) -> Iterable[Path]:
    yield from sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and not path.name.endswith(story_suffix)
            and not path.name.endswith(quiz_suffix)
        ),
        key=source_sort_key,
    )


def iter_story_markdown_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted((path for path in directory.glob(pattern) if path.is_file()), key=source_sort_key)


def iter_quiz_markdown_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted((path for path in directory.glob(pattern) if path.is_file()), key=source_sort_key)


def term_from_heading(text: str) -> str:
    match = TERM_RE.search(text)
    if not match:
        return ""
    return clean_inline_markdown(match.group("term"))


def strip_known_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    stem = path.stem
    if stem.endswith(".quiz"):
        return stem[: -len(".quiz")]
    return stem


def term_from_sibling_path(path: Path, *, suffix: str) -> str:
    base = strip_known_suffix(path, suffix)
    return filename_slug_term(Path(base))


def load_base_term_from_sibling(path: Path, *, suffix: str, kind: str) -> tuple[str, Counter[str]]:
    stats: Counter[str] = Counter()
    base = strip_known_suffix(path, suffix)
    base_path = path.with_name(f"{base}.md")
    fallback = term_from_sibling_path(path, suffix=suffix)
    if not base_path.is_file():
        stats[f"missing_base_md_for_{kind}"] += 1
        return fallback, stats
    try:
        text = base_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not read base Markdown for {path}: {exc}", file=sys.stderr)
        stats[f"unreadable_base_md_for_{kind}"] += 1
        return fallback, stats
    term = term_from_heading(text)
    if not term:
        stats[f"missing_base_term_header_for_{kind}"] += 1
        return fallback, stats
    return term, stats


def unique_terms(documents: Sequence[TextDocument]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc in documents:
        key = normalize_key(doc.term)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc.term)
    return out


def document_stats(documents: Sequence[TextDocument]) -> dict[str, object]:
    kind_counts = Counter(doc.kind for doc in documents)
    documents_per_term = Counter()
    chars_per_kind = Counter()
    for doc in documents:
        documents_per_term[normalize_key(doc.term)] += 1
        chars_per_kind[doc.kind] += len(doc.text)
    per_term_distribution = Counter(documents_per_term.values())
    return {
        "documents": len(documents),
        "terms": len(documents_per_term),
        "chars": sum(len(doc.text) for doc in documents),
        "documents_by_kind": {k: v for k, v in sorted(kind_counts.items())},
        "chars_by_kind": {k: v for k, v in sorted(chars_per_kind.items())},
        "documents_per_term": {str(k): v for k, v in sorted(per_term_distribution.items())},
    }


def load_midtrain_documents(
    source_dir: Path,
    source_glob: str,
    story_glob: str,
    *,
    story_suffix: str,
    quiz_suffix: str,
) -> tuple[list[TextDocument], dict[str, object]]:
    base_paths = list(
        iter_base_markdown_files(
            source_dir,
            source_glob,
            story_suffix=story_suffix,
            quiz_suffix=quiz_suffix,
        )
    )
    story_paths = list(iter_story_markdown_files(source_dir, story_glob))
    stats: Counter[str] = Counter(
        source_files=len(base_paths) + len(story_paths),
        base_md_files=len(base_paths),
        story_md_files=len(story_paths),
    )
    parsed_files = 0
    documents: list[TextDocument] = []
    for path in base_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"warning: skipping unreadable Markdown {path}: {exc}", file=sys.stderr)
            stats["unreadable_base_md_files"] += 1
            continue
        stats["source_chars"] += len(text)
        term = term_from_heading(text)
        if not term:
            stats["missing_term_header_files"] += 1
            term = filename_slug_term(path)
        elif not terms_match_filename(term, path):
            stats["filename_term_mismatches"] += 1
        body = normalize_term_markdown(text)
        if not body.strip():
            stats["empty_base_md_files"] += 1
            continue
        documents.append(TextDocument(term=term, kind="midtrain_md", text=body))
        parsed_files += 1
    for path in story_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"warning: skipping unreadable story Markdown {path}: {exc}", file=sys.stderr)
            stats["unreadable_story_md_files"] += 1
            continue
        stats["source_chars"] += len(text)
        body = normalize_story_text(text)
        if not body:
            stats["empty_story_md_files"] += 1
            continue
        term, term_stats = load_base_term_from_sibling(path, suffix=story_suffix, kind="story")
        stats.update(term_stats)
        documents.append(
            TextDocument(
                term=term,
                kind="midtrain_story",
                text=format_sibling_document(term, "Story", body),
            )
        )
        parsed_files += 1
    stats["parsed_files"] = parsed_files
    stats["skipped_files"] = stats["source_files"] - parsed_files
    source_stats: dict[str, object] = dict(stats)
    source_stats.update(document_stats(documents))
    source_stats["source_glob"] = source_glob
    source_stats["story_glob"] = story_glob
    source_stats["story_suffix_included"] = story_suffix
    source_stats["quiz_suffix_excluded"] = quiz_suffix
    return documents, source_stats


def load_sft_documents(
    source_dir: Path,
    quiz_glob: str,
    *,
    quiz_suffix: str,
) -> tuple[list[TextDocument], dict[str, object]]:
    paths = list(iter_quiz_markdown_files(source_dir, quiz_glob))
    stats: Counter[str] = Counter(source_files=len(paths))
    documents: list[TextDocument] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"warning: skipping unreadable quiz Markdown {path}: {exc}", file=sys.stderr)
            stats["unreadable_files"] += 1
            continue
        stats["source_chars"] += len(text)
        body = normalize_story_text(text)
        if not body:
            stats["empty_files"] += 1
            continue
        items = split_quiz_items(body)
        if not items:
            stats["empty_quiz_item_files"] += 1
            continue
        term, term_stats = load_base_term_from_sibling(path, suffix=quiz_suffix, kind="quiz")
        stats.update(term_stats)
        stats["quiz_items"] += len(items)
        stats["quiz_item_chars"] += sum(len(item) for item in items)
        for item in items:
            documents.append(
                TextDocument(
                    term=term,
                    kind="sft_quiz",
                    text=format_sibling_document(term, "Quiz", item),
                )
            )
    stats["parsed_files"] = sum(1 for path in paths if path.is_file()) - stats.get("unreadable_files", 0) - stats.get("empty_files", 0) - stats.get("empty_quiz_item_files", 0)
    stats["skipped_files"] = len(paths) - stats["parsed_files"]
    source_stats: dict[str, object] = dict(stats)
    source_stats.update(document_stats(documents))
    source_stats["quiz_glob"] = quiz_glob
    source_stats["quiz_suffix"] = quiz_suffix
    return documents, source_stats


def split_documents_by_term(
    documents: Sequence[TextDocument], val_fraction: float, split_seed: int
) -> tuple[list[TextDocument], list[TextDocument]]:
    if not documents:
        raise ValueError("no documents to split")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")

    grouped: dict[str, list[TextDocument]] = defaultdict(list)
    for doc in documents:
        grouped[normalize_key(doc.term)].append(doc)
    if len(grouped) == 1 or val_fraction == 0.0:
        return list(documents), []

    total_chars = sum(len(doc.text) for doc in documents)
    target_val_chars = max(1, int(round(total_chars * val_fraction)))
    groups = list(grouped.items())
    random.Random(split_seed).shuffle(groups)
    val_keys: set[str] = set()
    val_chars = 0
    for key, group in groups:
        if len(val_keys) >= len(grouped) - 1:
            break
        val_keys.add(key)
        val_chars += sum(len(doc.text) for doc in group)
        if val_chars >= target_val_chars:
            break
    train = [doc for doc in documents if normalize_key(doc.term) not in val_keys]
    val = [doc for doc in documents if normalize_key(doc.term) in val_keys]
    if not train:
        raise ValueError("split produced empty training set")
    return train, val


def write_jsonl_documents(path: Path, documents: Sequence[TextDocument]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc.text, "term": doc.term, "kind": doc.kind}, ensure_ascii=False))
            f.write("\n")


def format_documents(documents: Sequence[TextDocument]) -> str:
    text = "\n\n".join(doc.text for doc in documents)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def write_raw_split(
    out_dir: Path,
    train_documents: Sequence[TextDocument],
    val_documents: Sequence[TextDocument],
) -> DocumentSplit:
    raw_dir = out_dir / "raw_ita_dict_v3"
    ensure_clean_dir(raw_dir)
    train_path = raw_dir / "train.txt"
    val_path = raw_dir / "val.txt"
    train_jsonl_path = raw_dir / "train.jsonl"
    val_jsonl_path = raw_dir / "val.jsonl"
    train_text = format_documents(train_documents)
    val_text = format_documents(val_documents)
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    write_jsonl_documents(train_jsonl_path, train_documents)
    write_jsonl_documents(val_jsonl_path, val_documents)
    return DocumentSplit(
        train_terms=unique_terms(train_documents),
        val_terms=unique_terms(val_documents),
        train_chars=len(train_text),
        val_chars=len(val_text),
        train_documents=len(train_documents),
        val_documents=len(val_documents),
        train_sha256=file_sha256(train_path),
        val_sha256=file_sha256(val_path) if val_text else "",
        train_jsonl_sha256=file_sha256(train_jsonl_path),
        val_jsonl_sha256=file_sha256(val_jsonl_path) if val_documents else "",
    )


def load_tokenizer_info(path: Path) -> TokenizerInfo:
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("format") != "gd-bpe-tokenizer-v1":
        raise ValueError(f"unsupported tokenizer format in {path}: {spec.get('format')!r}")
    tokens = spec.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"tokenizer {path} has no tokens")
    vocab_size = int(spec.get("vocab_size") or len(tokens))
    max_id = max(int(token["id"]) for token in tokens if isinstance(token, Mapping) and "id" in token)
    if max_id + 1 > vocab_size:
        vocab_size = max_id + 1
    if vocab_size > 0xFFFF:
        raise ValueError(
            f"tokenizer vocab_size={vocab_size} does not fit compact uint16 storage"
        )
    return TokenizerInfo(
        path=path,
        vocab_size=vocab_size,
        sha256=file_sha256(path),
        spec_hash=str(spec.get("hash", "")),
        pad_id=tokenizer_special_id(path, PAD),
        im_start_id=tokenizer_special_id(path, IM_START),
        im_end_id=tokenizer_special_id(path, IM_END),
    )


def copy_tokenizer_for_dataset(tokenizer: TokenizerInfo, out_dir: Path) -> Path:
    dst = out_dir / f"tokenizer-v{tokenizer.vocab_size}.json"
    if tokenizer.path.resolve() == dst.resolve():
        return dst
    shutil.copy2(tokenizer.path, dst)
    return dst


def write_manifest(
    *,
    kind: str,
    objective: str,
    out_dir: Path,
    source_dir: Path,
    source_pattern: str,
    tokenizer: TokenizerInfo,
    local_tokenizer_path: Path,
    split: DocumentSplit,
    source_stats: Mapping[str, object],
    train_corpus: TokenizedCorpus,
    val_corpus: TokenizedCorpus | None,
    gdds_paths: Mapping[str, Sequence[Path]],
    max_shard_bytes: int,
    context_length: int,
    val_fraction: float,
    split_seed: int,
) -> None:
    storage_fields = storage_fields_for_context(context_length)
    runtime_fields = fields_for_context(context_length)
    split_samples = {
        split_name: sum(read_gdds_header(path).samples for path in paths)
        for split_name, paths in gdds_paths.items()
    }
    val_records = 0 if val_corpus is None else val_corpus.num_sequences
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    manifest = {
        "format": "GDDS",
        "version": 1,
        "dataset_builder": Path(__file__).name,
        "dataset_kind": kind,
        "objective": objective,
        "schema_hash": f"0x{schema_hash(storage_fields):016x}",
        "fields": [field_metadata(field_spec) for field_spec in storage_fields],
        "runtime_schema_hash": f"0x{schema_hash(runtime_fields):016x}",
        "runtime_fields": [field_metadata(field_spec) for field_spec in runtime_fields],
        "storage": {
            "tokens": f"per-sample uint16 token ids, fixed length {context_length + 1}; runtime transform emits shifted i32 inputs/targets",
            "segment_lengths": "per-row document-fragment lengths over tokens[:-1], summing to context_length",
            "target_masking": "compact GPT-LM transform masks only cross-document targets using pad_id",
        },
        "tokenizer": {
            "path": local_tokenizer_path.name,
            "source_path": str(tokenizer.path),
            "source_sha256": tokenizer.sha256,
            "tokenizer_hash": train_corpus.manifest.get("tokenizer_hash", tokenizer.spec_hash),
            "spec_hash": tokenizer.spec_hash,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "im_start_id": tokenizer.im_start_id,
            "im_end_id": tokenizer.im_end_id,
            "mode": "pretrained_existing_tokenizer",
        },
        "packing": {
            "record_length": context_length + 1,
            "context_length": context_length,
            "stride": context_length,
            "source_mode": "jsonl_per_document_with_im_delimiters",
            "source_records": train_corpus.num_sequences + val_records,
            "source_tokens": train_corpus.num_tokens + val_tokens,
            "train_source_tokens": train_corpus.num_tokens,
            "val_source_tokens": val_tokens,
            "train_records": split_samples.get("train", 0),
            "val_records": split_samples.get("val", 0),
            "train_packed_tokens": split_samples.get("train", 0) * context_length,
            "val_packed_tokens": split_samples.get("val", 0) * context_length,
            "train_source_records": train_corpus.num_sequences,
            "val_source_records": val_records,
        },
        "splits": {
            split_name: {
                "samples": split_samples.get(split_name, 0),
                "shards": [path.name for path in paths],
            }
            for split_name, paths in gdds_paths.items()
        },
        "raw_split": {
            "mode": "term_random_before_tokenization",
            "val_fraction": val_fraction,
            "split_seed": split_seed,
            "train_terms": len(split.train_terms),
            "val_terms": len(split.val_terms),
            "train_documents": split.train_documents,
            "val_documents": split.val_documents,
            "train_chars": split.train_chars,
            "val_chars": split.val_chars,
            "train_sha256": split.train_sha256,
            "val_sha256": split.val_sha256,
            "train_jsonl_sha256": split.train_jsonl_sha256,
            "val_jsonl_sha256": split.val_jsonl_sha256,
            "train_path": "raw_ita_dict_v3/train.txt",
            "val_path": "raw_ita_dict_v3/val.txt",
            "train_jsonl_path": "raw_ita_dict_v3/train.jsonl",
            "val_jsonl_path": "raw_ita_dict_v3/val.jsonl",
        },
        "dictionary_format": {
            "midtrain": "base *.md dictionary pages plus *.story.md siblings; .quiz.md siblings excluded",
            "sft": "*.quiz.md pages only; split on Markdown horizontal rules (`---`) into one Q&A document per block",
            "sequence_delimiters": "tokenizer inserts <|im_start|>/<|im_end|> around every JSONL row before stream packing",
        },
        "source": {
            "source_dir": str(source_dir),
            "source_pattern": source_pattern,
            **dict(source_stats),
        },
        "prep": {
            "format_version": DATA_FORMAT_VERSION,
            "max_shard_bytes": max_shard_bytes,
            "attention_boundaries": "document-local segments inside each fixed row via stored segment_lengths",
            "ignore_index": tokenizer.pad_id,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "split_terms.json").write_text(
        json.dumps(
            {"train": split.train_terms, "val": split.val_terms},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_dataset(
    *,
    kind: str,
    objective: str,
    documents: Sequence[TextDocument],
    source_stats: Mapping[str, object],
    source_dir: Path,
    source_pattern: str,
    out_dir: Path,
    tokenizer: TokenizerInfo,
    context_length: int,
    max_shard_bytes: int,
    val_fraction: float,
    split_seed: int,
    preview_samples: int,
    preview_seed: int,
) -> DatasetSummary:
    if not documents:
        raise ValueError(f"no {kind} documents loaded")
    resolved_out = out_dir.resolve()
    resolved_tokenizer = tokenizer.path.resolve()
    if resolved_tokenizer == resolved_out or path_is_relative_to(resolved_tokenizer, resolved_out):
        raise ValueError(
            f"--tokenizer ({tokenizer.path}) must not be inside output dir {out_dir}; "
            "the output dir is cleaned before writing"
        )

    train_docs, val_docs = split_documents_by_term(documents, val_fraction, split_seed)
    ensure_clean_dir(out_dir)
    local_tokenizer_path = copy_tokenizer_for_dataset(tokenizer, out_dir)
    split = write_raw_split(out_dir, train_docs, val_docs)
    print(
        f"{kind}: raw split: train_terms={len(split.train_terms)} val_terms={len(split.val_terms)} "
        f"train_documents={split.train_documents} val_documents={split.val_documents} "
        f"train_chars={split.train_chars} val_chars={split.val_chars} "
        f"val_fraction={val_fraction:.3f} seed={split_seed}"
    )

    train_corpus = prepare_tokenized_corpus(
        split_name=f"{kind}/train",
        input_path=out_dir / "raw_ita_dict_v3" / "train.jsonl",
        output_dir=out_dir / "tokenized_train",
        tokenizer_path=local_tokenizer_path,
        vocab_size=tokenizer.vocab_size,
        min_frequency=1,
        use_existing_tokenizer=True,
    )
    val_corpus = None
    if val_docs:
        val_corpus = prepare_tokenized_corpus(
            split_name=f"{kind}/val",
            input_path=out_dir / "raw_ita_dict_v3" / "val.jsonl",
            output_dir=out_dir / "tokenized_val",
            tokenizer_path=local_tokenizer_path,
            vocab_size=tokenizer.vocab_size,
            min_frequency=1,
            use_existing_tokenizer=True,
        )

    paths = write_gdds_dataset(
        out_dir=out_dir,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        context_length=context_length,
        max_shard_bytes=max_shard_bytes,
    )
    write_manifest(
        kind=kind,
        objective=objective,
        out_dir=out_dir,
        source_dir=source_dir,
        source_pattern=source_pattern,
        tokenizer=tokenizer,
        local_tokenizer_path=local_tokenizer_path,
        split=split,
        source_stats=source_stats,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        gdds_paths=paths,
        max_shard_bytes=max_shard_bytes,
        context_length=context_length,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    train_samples = sum(read_gdds_header(path).samples for path in paths["train"])
    val_samples = sum(read_gdds_header(path).samples for path in paths.get("val", []))
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    print(
        f"{kind}: wrote {train_samples + val_samples} GPT LM samples to {out_dir} "
        f"(train={train_samples}, val={val_samples})"
    )
    print(
        f"{kind}: token counts with delimiters: train={train_corpus.num_tokens} "
        f"val={val_tokens} total={train_corpus.num_tokens + val_tokens}"
    )
    for split_paths in paths.values():
        for path in split_paths:
            print(path)
    print(out_dir / "manifest.json")
    show_random_samples(
        paths["train"],
        local_tokenizer_path,
        train_samples,
        count=preview_samples,
        seed=preview_seed,
    )
    return DatasetSummary(
        kind=kind,
        out_dir=out_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        train_documents=split.train_documents,
        val_documents=split.val_documents,
        train_tokens=train_corpus.num_tokens,
        val_tokens=val_tokens,
    )


def write_parent_manifest(out_dir: Path, tokenizer: TokenizerInfo, summaries: Sequence[DatasetSummary]) -> None:
    if not summaries:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "gpt-lm-ita-dict-v3-bundle",
        "format_version": DATA_FORMAT_VERSION,
        "tokenizer": {
            "source_path": str(tokenizer.path),
            "source_sha256": tokenizer.sha256,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "im_start_id": tokenizer.im_start_id,
            "im_end_id": tokenizer.im_end_id,
        },
        "datasets": [
            {
                "kind": summary.kind,
                "path": str(summary.out_dir),
                "train_samples": summary.train_samples,
                "val_samples": summary.val_samples,
                "train_documents": summary.train_documents,
                "val_documents": summary.val_documents,
                "train_tokens": summary.train_tokens,
                "val_tokens": summary.val_tokens,
            }
            for summary in summaries
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--source-glob",
        default=DEFAULT_SOURCE_GLOB,
        help="glob for base mid-training Markdown files; .story/.quiz siblings are excluded from this glob",
    )
    parser.add_argument("--story-glob", default=DEFAULT_STORY_GLOB)
    parser.add_argument("--quiz-glob", default=DEFAULT_QUIZ_GLOB)
    parser.add_argument("--story-suffix", default=DEFAULT_STORY_SUFFIX)
    parser.add_argument("--quiz-suffix", default=DEFAULT_QUIZ_SUFFIX)
    parser.add_argument("--tokenizer", type=Path, required=True, help="pre-trained gd-bpe tokenizer JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="parent directory for midtrain/ and sft/")
    parser.add_argument("--midtrain-out-dir", type=Path, default=None)
    parser.add_argument("--sft-out-dir", type=Path, default=None)
    parser.add_argument("--only", choices=("both", "midtrain", "sft"), default="both")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--sft-split-seed",
        type=int,
        default=None,
        help="optional distinct split seed for quiz/SFT data; defaults to --split-seed",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--preview-seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    tokenizer_path = args.tokenizer.expanduser()
    out_dir = args.out_dir.expanduser()
    midtrain_out_dir = (
        args.midtrain_out_dir.expanduser()
        if args.midtrain_out_dir is not None
        else out_dir / DEFAULT_MIDTRAIN_SUBDIR
    )
    sft_out_dir = (
        args.sft_out_dir.expanduser()
        if args.sft_out_dir is not None
        else out_dir / DEFAULT_SFT_SUBDIR
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1")
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")

    tokenizer = load_tokenizer_info(tokenizer_path)
    print(
        "Tokenizer: "
        f"path={tokenizer.path} vocab_size={tokenizer.vocab_size} "
        f"pad_id={tokenizer.pad_id} im_start_id={tokenizer.im_start_id} im_end_id={tokenizer.im_end_id}"
    )

    summaries: list[DatasetSummary] = []
    if args.only in ("both", "midtrain"):
        mid_docs, mid_stats = load_midtrain_documents(
            source_dir,
            args.source_glob,
            args.story_glob,
            story_suffix=args.story_suffix,
            quiz_suffix=args.quiz_suffix,
        )
        print(
            "Loaded midtrain Markdown: "
            f"source_files={mid_stats.get('source_files', 0)} documents={len(mid_docs)} "
            f"terms={mid_stats.get('terms', 0)} chars={mid_stats.get('chars', 0)}"
        )
        summaries.append(
            build_dataset(
                kind="midtrain",
                objective="causal_lm_midtraining_base_and_story_markdown",
                documents=mid_docs,
                source_stats=mid_stats,
                source_dir=source_dir,
                source_pattern=args.source_glob,
                out_dir=midtrain_out_dir,
                tokenizer=tokenizer,
                context_length=args.context_length,
                max_shard_bytes=args.max_shard_bytes,
                val_fraction=args.val_fraction,
                split_seed=args.split_seed,
                preview_samples=args.preview_samples,
                preview_seed=args.preview_seed,
            )
        )

    if args.only in ("both", "sft"):
        sft_docs, sft_stats = load_sft_documents(
            source_dir,
            args.quiz_glob,
            quiz_suffix=args.quiz_suffix,
        )
        print(
            "Loaded SFT quiz Markdown: "
            f"source_files={sft_stats.get('source_files', 0)} documents={len(sft_docs)} "
            f"terms={sft_stats.get('terms', 0)} chars={sft_stats.get('chars', 0)} "
            f"quiz_items={sft_stats.get('quiz_items', 0)}"
        )
        summaries.append(
            build_dataset(
                kind="sft",
                objective="causal_lm_sft_quiz_markdown",
                documents=sft_docs,
                source_stats=sft_stats,
                source_dir=source_dir,
                source_pattern=args.quiz_glob,
                out_dir=sft_out_dir,
                tokenizer=tokenizer,
                context_length=args.context_length,
                max_shard_bytes=args.max_shard_bytes,
                val_fraction=args.val_fraction,
                split_seed=args.sft_split_seed if args.sft_split_seed is not None else args.split_seed,
                preview_samples=args.preview_samples,
                preview_seed=args.preview_seed,
            )
        )

    write_parent_manifest(out_dir, tokenizer, summaries)
    print("dataset_ita_dict_v3: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
