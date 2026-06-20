#!/usr/bin/env python3
"""Prepare a packed GPT LM dataset from Gemma-cleaned Italian dictionary Markdown.

The source directory contains one Markdown wiki page per generated Italian term,
for example::

    # Termine: moltiplicare
    ## Definizioni
    ### 1. Eseguire una moltiplicazione
    ...
    **Parte del discorso:** verbo
    **Esempi:**
    * ...

This builder parses the term heading and definition metadata for statistics and
term-level splitting, emits each source term Markdown page as a standalone
document with its original Markdown structure except for removing the leading
``# `` from the term header, loads optional ``.story.md`` siblings as separate
documents, splits optional ``.quiz.md`` siblings into one document per
``---``-separated Q&A, splits by term before tokenizer training,
tokenizes one JSONL row per document with
``<|im_start|>``/``<|im_end|>`` delimiters, then repacks the delimited token
stream into compact fixed-size GPT LM samples.  On disk, each GDDS sample stores
``uint16`` token rows plus small segment-length metadata; the C dataloader
transform expands those rows into the runtime ``input_ids``/``target_ids`` /
``positions``/``cu_seqlens`` tensors consumed by ``examples/gpt_lm``.
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
import unicodedata
from collections import Counter, defaultdict
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
from tok_utils import TOKEN_DTYPE, TokenizedCorpus, tokenize_jsonl  # noqa: E402

PAD = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SOURCE_DIR = Path(
    "/Users/mascit/projects/DataFarmer/data/definizioni-clean-gemma"
)
DEFAULT_SOURCE_GLOB = "*.md"
DEFAULT_STORY_SUFFIX = ".story.md"
DEFAULT_QUIZ_SUFFIX = ".quiz.md"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_MAX_EXAMPLES_PER_DEFINITION = -1  # no limit
MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
DATA_FORMAT_VERSION = "gpt-lm-ita-dict-v2-md-gemma-docs-v6-compact-u16-segments"

TERM_RE = re.compile(r"^#\s+Termine:\s*(?P<term>.+?)\s*$", re.MULTILINE)
DEFINITION_HEADING_RE = re.compile(
    r"^###\s+(?P<number>\d+)\.\s+(?P<title>.+?)\s*$", re.MULTILINE
)
FIELD_RE = re.compile(r"^\*\*(?P<label>[^*\n]+?):\*\*\s*(?P<value>.*?)\s*$")
BULLET_RE = re.compile(r"^\s*\*\s+(?P<text>.+?)\s*$")
SOURCE_NAME_RE = re.compile(r"^(?P<id>\d+)-(?P<slug>.+)$")
EXPECTED_FIELDS = (
    "Parte del discorso",
    "Concetti chiave",
    "Ambiti",
    "Esempi",
    "Sinonimi",
    "Correlati",
    "Contrari",
)
NONE_VALUES = {"nessuno", "nessuna", "nessun", "nessun sinonimo", "nessun contrario"}


@dataclass(frozen=True)
class ShardHeader:
    samples: int
    index_offset: int
    schema_hash: int


@dataclass
class DefinitionEntry:
    title: str
    definition: str
    part_of_speech: str = ""
    key_concepts: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    antonyms: list[str] = field(default_factory=list)


@dataclass
class TermEntry:
    term: str
    term_markdown: str = ""
    definitions: list[DefinitionEntry] = field(default_factory=list)
    stories: list[str] = field(default_factory=list)
    quizzes: list[str] = field(default_factory=list)
    source_files: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CorpusSplit:
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
class TextDocument:
    term: str
    kind: str
    text: str


def tokenizer_special_id(tokenizer_path: Path, text: str) -> int:
    with tokenizer_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    for token in spec.get("tokens", []):
        if isinstance(token, Mapping) and token.get("kind") == "special" and token.get("text") == text:
            return int(token["id"])
    raise ValueError(f"special token {text!r} not found in {tokenizer_path}")


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
    """Runtime fields exposed by the compact GDDS transform."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return [
        FieldSpec("input_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec("positions", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec("target_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec(
            "segment_lengths", "i32", (-1,), collate="packed_sequence", ragged_dim=0
        ),
        FieldSpec(
            "cu_seqlens",
            "i32",
            (-1,),
            collate="generated",
            generated="cu_seqlens_from_lengths",
            source="segment_lengths",
        ),
    ]


def storage_fields_for_context(context_length: int) -> list[FieldSpec]:
    """Compact on-disk fields.

    Each sample stores one uint16 token row of length ``context_length + 1``.
    Runtime ``input_ids``/``target_ids`` are derived by shifting this row.  We
    also store per-row document-fragment lengths so positions/cu_seqlens can be
    reconstructed exactly without re-running document-id logic during training.
    """

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return [
        FieldSpec("tokens", "u16", (context_length + 1,), collate="stack"),
        FieldSpec(
            "segment_lengths",
            "i32",
            (-1,),
            collate="packed_sequence",
            ragged_dim=0,
        ),
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


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_inline_markdown(text: object) -> str:
    text = normalize_space(text)
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\\([*_`\[\]()#+.!-])", r"\1", text)
    return normalize_space(text)


def normalize_term_markdown(text: object) -> str:
    """Preserve a term Markdown document, but make it start with ``Termine:``."""

    if not isinstance(text, str):
        return ""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\A#\s+(?=Termine:)", "", text, count=1)


def normalize_story_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_none_value(text: str) -> bool:
    return normalize_key(text).strip(" .;:") in NONE_VALUES


def dedupe_texts(texts: Iterable[str], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        text = clean_inline_markdown(text)
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


def split_list_field(value: str, *, drop_none_fields: bool) -> list[str]:
    value = clean_inline_markdown(value)
    if not value:
        return []
    if drop_none_fields and is_none_value(value):
        return []
    return dedupe_texts(part.strip(" .;") for part in value.split(","))


def source_sort_key(path: Path) -> tuple[int, str]:
    match = SOURCE_NAME_RE.match(path.stem)
    if match:
        return int(match.group("id")), path.name
    return 2_147_483_647, path.name


def iter_markdown_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and not path.name.endswith(DEFAULT_STORY_SUFFIX)
            and not path.name.endswith(DEFAULT_QUIZ_SUFFIX)
        ),
        key=source_sort_key,
    )


def filename_slug_term(path: Path) -> str:
    match = SOURCE_NAME_RE.match(path.stem)
    slug = match.group("slug") if match else path.stem
    return slug.replace("-", " ")


def terms_match_filename(term: str, path: Path) -> bool:
    lhs = strip_accents(normalize_key(term).replace("-", " ")).lstrip("'")
    rhs = strip_accents(normalize_key(filename_slug_term(path)).replace("-", " ")).lstrip("'")
    return lhs == rhs


def extract_definition_paragraph(block: str) -> str:
    paragraph: list[str] = []
    in_paragraph = False
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_paragraph:
                break
            continue
        if stripped == "---" or stripped.startswith("**"):
            break
        paragraph.append(clean_inline_markdown(stripped))
        in_paragraph = True
    return normalize_space(" ".join(paragraph))


def parse_definition_block(
    *,
    title: str,
    block: str,
    max_examples_per_definition: int,
    drop_none_fields: bool,
) -> tuple[DefinitionEntry | None, Counter[str]]:
    stats: Counter[str] = Counter()
    definition = extract_definition_paragraph(block)
    fields: dict[str, str] = {}
    examples: list[str] = []
    collecting_examples = False

    for raw_line in block.splitlines():
        line = raw_line.strip()
        field_match = FIELD_RE.match(line)
        if field_match:
            label = clean_inline_markdown(field_match.group("label"))
            value = clean_inline_markdown(field_match.group("value"))
            fields[label] = value
            collecting_examples = label == "Esempi"
            continue
        if collecting_examples:
            bullet_match = BULLET_RE.match(raw_line)
            if bullet_match:
                examples.append(clean_inline_markdown(bullet_match.group("text")))

    present_fields = set(fields)
    for label in EXPECTED_FIELDS:
        if label not in present_fields:
            stats[f"missing_field_{label}"] += 1
    for label in ("Sinonimi", "Contrari"):
        if is_none_value(fields.get(label, "")):
            stats[f"none_{label}"] += 1
            if drop_none_fields:
                stats[f"dropped_none_{label}"] += 1

    title = clean_inline_markdown(title)
    if not title and not definition:
        return None, stats
    if not definition:
        stats["missing_definition_paragraph"] += 1

    entry = DefinitionEntry(
        title=title,
        definition=definition,
        part_of_speech=clean_inline_markdown(fields.get("Parte del discorso", "")),
        key_concepts=split_list_field(
            fields.get("Concetti chiave", ""), drop_none_fields=drop_none_fields
        ),
        domains=split_list_field(fields.get("Ambiti", ""), drop_none_fields=drop_none_fields),
        examples=dedupe_texts(
            examples,
            max_items=max_examples_per_definition,
        ),
        synonyms=split_list_field(
            fields.get("Sinonimi", ""), drop_none_fields=drop_none_fields
        ),
        related=split_list_field(
            fields.get("Correlati", ""), drop_none_fields=drop_none_fields
        ),
        antonyms=split_list_field(
            fields.get("Contrari", ""), drop_none_fields=drop_none_fields
        ),
    )
    if max_examples_per_definition >= 0 and len(examples) > len(entry.examples):
        stats["truncated_examples"] += len(examples) - len(entry.examples)
    return entry, stats


def split_quiz_items(text: str) -> list[str]:
    return [
        item
        for item in (normalize_story_text(part) for part in re.split(r"(?m)^\s*---\s*$", text))
        if item
    ]


def read_text_sibling(path: Path, suffix: str, kind: str) -> tuple[list[str], Counter[str]]:
    stats: Counter[str] = Counter()
    sibling_path = path.with_name(f"{path.stem}{suffix}")
    if not sibling_path.is_file():
        stats[f"missing_{kind}_files"] += 1
        return [], stats
    try:
        text = sibling_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: skipping unreadable {kind} {sibling_path}: {exc}", file=sys.stderr)
        stats[f"unreadable_{kind}_files"] += 1
        return [], stats
    body = normalize_story_text(text)
    if not body:
        stats[f"empty_{kind}_files"] += 1
        return [], stats
    documents = split_quiz_items(body) if kind == "quiz" else [body]
    if not documents:
        stats[f"empty_{kind}_files"] += 1
        return [], stats
    stats[f"{kind}_files"] += 1
    stats[f"{kind}s"] += len(documents)
    stats[f"{kind}_chars"] += sum(len(document) for document in documents)
    return documents, stats


def parse_markdown_file(
    path: Path,
    *,
    max_examples_per_definition: int,
    drop_none_fields: bool,
    story_suffix: str,
    quiz_suffix: str,
) -> tuple[TermEntry | None, Counter[str]]:
    stats: Counter[str] = Counter()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: skipping unreadable Markdown {path}: {exc}", file=sys.stderr)
        stats["unreadable_files"] += 1
        return None, stats

    stats["source_chars"] += len(text)
    term_match = TERM_RE.search(text)
    if not term_match:
        print(f"warning: skipping {path}: missing '# Termine:' heading", file=sys.stderr)
        stats["missing_term_header_files"] += 1
        return None, stats
    term = clean_inline_markdown(term_match.group("term"))
    if not term:
        print(f"warning: skipping {path}: empty term heading", file=sys.stderr)
        stats["empty_term_header_files"] += 1
        return None, stats
    if not terms_match_filename(term, path):
        stats["filename_term_mismatches"] += 1
    if "## Definizioni" not in text:
        stats["missing_definitions_header_files"] += 1

    headings = list(DEFINITION_HEADING_RE.finditer(text))
    if not headings:
        print(f"warning: skipping {path}: no definition headings", file=sys.stderr)
        stats["no_definition_heading_files"] += 1
        return None, stats
    numbers = [int(match.group("number")) for match in headings]
    if numbers != list(range(1, len(numbers) + 1)):
        stats["non_sequential_numbering_files"] += 1

    definitions: list[DefinitionEntry] = []
    for index, heading in enumerate(headings):
        block_start = heading.end()
        block_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        defn, def_stats = parse_definition_block(
            title=heading.group("title"),
            block=text[block_start:block_end],
            max_examples_per_definition=max_examples_per_definition,
            drop_none_fields=drop_none_fields,
        )
        stats.update(def_stats)
        if defn is not None:
            definitions.append(defn)

    if not definitions:
        stats["empty_definition_files"] += 1
        return None, stats
    stories, story_stats = read_text_sibling(path, story_suffix, "story")
    quizzes, quiz_stats = read_text_sibling(path, quiz_suffix, "quiz")
    stats.update(story_stats)
    stats.update(quiz_stats)
    stats["raw_definitions"] += len(definitions)
    return (
        TermEntry(
            term=term,
            term_markdown=normalize_term_markdown(text),
            definitions=definitions,
            stories=stories,
            quizzes=quizzes,
            source_files={path.name},
        ),
        stats,
    )


def merge_list(existing: list[str], incoming: list[str], *, max_items: int = -1) -> list[str]:
    return dedupe_texts([*existing, *incoming], max_items=max_items)


def merge_stories(existing: list[str], incoming: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for story in [*existing, *incoming]:
        story = normalize_story_text(story)
        if not story:
            continue
        key = normalize_key(story)
        if key in seen:
            continue
        seen.add(key)
        out.append(story)
    return out


def merge_definition_entry(
    target: DefinitionEntry,
    source: DefinitionEntry,
    max_examples_per_definition: int,
) -> None:
    if source.title and not target.title:
        target.title = source.title
    if source.definition and (
        not target.definition or len(source.definition) > len(target.definition)
    ):
        target.definition = source.definition
    if source.part_of_speech and not target.part_of_speech:
        target.part_of_speech = source.part_of_speech
    target.key_concepts = merge_list(target.key_concepts, source.key_concepts)
    target.domains = merge_list(target.domains, source.domains)
    target.examples = merge_list(
        target.examples, source.examples, max_items=max_examples_per_definition
    )
    target.synonyms = merge_list(target.synonyms, source.synonyms)
    target.related = merge_list(target.related, source.related)
    target.antonyms = merge_list(target.antonyms, source.antonyms)


def merge_term_entry(
    target: TermEntry,
    source: TermEntry,
    max_examples_per_definition: int,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    target.source_files.update(source.source_files)
    before_stories = len(target.stories)
    target.stories = merge_stories(target.stories, source.stories)
    stats["merged_story_duplicates"] += before_stories + len(source.stories) - len(target.stories)
    before_quizzes = len(target.quizzes)
    target.quizzes = merge_stories(target.quizzes, source.quizzes)
    stats["merged_quiz_duplicates"] += before_quizzes + len(source.quizzes) - len(target.quizzes)
    by_definition = {
        normalize_key(defn.definition): defn
        for defn in target.definitions
        if defn.definition
    }
    by_title = {normalize_key(defn.title): defn for defn in target.definitions if defn.title}
    for defn in source.definitions:
        existing = None
        definition_key = normalize_key(defn.definition)
        title_key = normalize_key(defn.title)
        if definition_key:
            existing = by_definition.get(definition_key)
        if existing is None and title_key:
            existing = by_title.get(title_key)
        if existing is None:
            target.definitions.append(defn)
            if definition_key:
                by_definition[definition_key] = defn
            if title_key:
                by_title[title_key] = defn
            continue
        merge_definition_entry(existing, defn, max_examples_per_definition)
        stats["merged_duplicate_definition_senses"] += 1
    return stats


def count_entry_stats(entries: Sequence[TermEntry]) -> dict[str, int]:
    definitions = [defn for entry in entries for defn in entry.definitions]
    return {
        "terms": len(entries),
        "definitions": len(definitions),
        "examples": sum(len(defn.examples) for defn in definitions),
        "synonyms": sum(len(defn.synonyms) for defn in definitions),
        "related_terms": sum(len(defn.related) for defn in definitions),
        "antonyms": sum(len(defn.antonyms) for defn in definitions),
        "merged_stories": sum(len(entry.stories) for entry in entries),
        "merged_story_chars": sum(len(story) for entry in entries for story in entry.stories),
        "merged_quizzes": sum(len(entry.quizzes) for entry in entries),
        "merged_quiz_chars": sum(len(quiz) for entry in entries for quiz in entry.quizzes),
    }


def load_entries(
    source_dir: Path,
    source_glob: str,
    max_examples_per_definition: int,
    drop_none_fields: bool,
    story_suffix: str,
    quiz_suffix: str,
) -> tuple[list[TermEntry], dict[str, object]]:
    paths = list(iter_markdown_files(source_dir, source_glob))
    stats: Counter[str] = Counter(source_files=len(paths))
    raw_entries: list[TermEntry] = []
    for path in paths:
        entry, file_stats = parse_markdown_file(
            path,
            max_examples_per_definition=max_examples_per_definition,
            drop_none_fields=drop_none_fields,
            story_suffix=story_suffix,
            quiz_suffix=quiz_suffix,
        )
        stats.update(file_stats)
        if entry is not None:
            raw_entries.append(entry)

    grouped: dict[str, list[TermEntry]] = defaultdict(list)
    for entry in raw_entries:
        grouped[normalize_key(entry.term)].append(entry)

    entries = sorted(
        raw_entries,
        key=lambda entry: (
            normalize_key(entry.term),
            sorted(entry.source_files)[0] if entry.source_files else "",
        ),
    )
    raw_duplicate_groups = sum(1 for group in grouped.values() if len(group) > 1)
    raw_duplicate_files = sum(len(group) - 1 for group in grouped.values() if len(group) > 1)
    entry_stats = count_entry_stats(entries)
    entry_stats["source_term_entries"] = len(entries)
    entry_stats["terms"] = len(grouped)
    stats.update(
        {
            "parsed_files": len(raw_entries),
            "skipped_files": len(paths) - len(raw_entries),
            "raw_terms": len(raw_entries),
            "duplicate_term_groups": raw_duplicate_groups,
            "duplicate_term_extra_files": raw_duplicate_files,
            **entry_stats,
        }
    )
    docs = documents_for_entries(entries)
    doc_kind_counts = Counter(doc.kind for doc in docs)
    stats["formatted_chars"] = sum(len(doc.text) for doc in docs)
    stats["documents"] = len(docs)
    stats["term_documents"] = doc_kind_counts.get("term", 0)
    stats["story_documents"] = doc_kind_counts.get("story", 0)
    stats["quiz_documents"] = doc_kind_counts.get("quiz", 0)

    # Keep the manifest compact but preserve the most useful distributional facts.
    definitions_per_term_document = Counter(len(entry.definitions) for entry in entries)
    stories_per_term = Counter(
        sum(len(entry.stories) for entry in group) for group in grouped.values()
    )
    quizzes_per_term = Counter(
        sum(len(entry.quizzes) for entry in group) for group in grouped.values()
    )
    documents_per_term = Counter(
        sum(1 + len(entry.stories) + len(entry.quizzes) for entry in group)
        for group in grouped.values()
    )
    source_files_per_term = Counter(len(group) for group in grouped.values())
    source_stats: dict[str, object] = dict(stats)
    source_stats["definitions_per_term_document"] = {
        str(k): v for k, v in sorted(definitions_per_term_document.items())
    }
    source_stats["documents_per_term"] = {
        str(k): v for k, v in sorted(documents_per_term.items())
    }
    source_stats["stories_per_term"] = {
        str(k): v for k, v in sorted(stories_per_term.items())
    }
    source_stats["quizzes_per_term"] = {
        str(k): v for k, v in sorted(quizzes_per_term.items())
    }
    source_stats["source_files_per_term"] = {
        str(k): v for k, v in sorted(source_files_per_term.items())
    }
    source_stats["drop_none_fields"] = drop_none_fields
    source_stats["story_suffix"] = story_suffix
    source_stats["quiz_suffix"] = quiz_suffix
    return entries, source_stats


def append_list_line(lines: list[str], label: str, values: Sequence[str]) -> None:
    if values:
        lines.append(f"   {label}: {', '.join(values)}")


def format_entry(entry: TermEntry) -> str:
    """Return the standalone term/definition document, preserving source Markdown."""

    if entry.term_markdown:
        return entry.term_markdown

    lines = [f"Termine: {entry.term}", "Definizioni:"]
    for index, definition in enumerate(entry.definitions, start=1):
        title = definition.title or "Definizione"
        lines.append(f"{index}. {title}")
        if definition.definition:
            lines.append(f"   Definizione: {definition.definition}")
        if definition.part_of_speech:
            lines.append(f"   Parte del discorso: {definition.part_of_speech}")
        append_list_line(lines, "Concetti chiave", definition.key_concepts)
        append_list_line(lines, "Ambiti", definition.domains)
        if definition.examples:
            lines.append("   Esempi:")
            for example in definition.examples:
                lines.append(f"   - {example}")
        append_list_line(lines, "Sinonimi", definition.synonyms)
        append_list_line(lines, "Correlati", definition.related)
        append_list_line(lines, "Contrari", definition.antonyms)
    return "\n".join(lines)


def format_sibling_document(term: str, section: str, body: str) -> str:
    return f"Termine: {term}\n\n## {section}\n{body}"


def iter_entry_documents(entry: TermEntry) -> Iterable[TextDocument]:
    yield TextDocument(term=entry.term, kind="term", text=format_entry(entry))
    for story in entry.stories:
        yield TextDocument(
            term=entry.term,
            kind="story",
            text=format_sibling_document(entry.term, "Story", story),
        )
    for quiz in entry.quizzes:
        yield TextDocument(
            term=entry.term,
            kind="quiz",
            text=format_sibling_document(entry.term, "Quiz", quiz),
        )


def documents_for_entries(entries: Sequence[TermEntry]) -> list[TextDocument]:
    return [doc for entry in entries for doc in iter_entry_documents(entry)]


def entry_chars(entry: TermEntry) -> int:
    return sum(len(doc.text) for doc in iter_entry_documents(entry))


def split_entries(
    entries: Sequence[TermEntry], val_fraction: float, split_seed: int
) -> tuple[list[TermEntry], list[TermEntry]]:
    if not entries:
        raise ValueError("no entries to split")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")

    grouped: dict[str, list[TermEntry]] = defaultdict(list)
    for entry in entries:
        key = normalize_key(entry.term)
        grouped[key].append(entry)
    if len(grouped) == 1 or val_fraction == 0.0:
        return list(entries), []

    total_chars = sum(entry_chars(entry) for entry in entries)
    target_val_chars = max(1, int(round(total_chars * val_fraction)))
    groups = list(grouped.items())
    random.Random(split_seed).shuffle(groups)
    val_keys: set[str] = set()
    val_chars = 0
    for key, group in groups:
        if len(val_keys) >= len(grouped) - 1:
            break
        val_keys.add(key)
        val_chars += sum(entry_chars(entry) for entry in group)
        if val_chars >= target_val_chars:
            break
    train = [entry for entry in entries if normalize_key(entry.term) not in val_keys]
    val = [entry for entry in entries if normalize_key(entry.term) in val_keys]
    if not train:
        raise ValueError("split produced empty training set")
    return train, val


def unique_terms(entries: Sequence[TermEntry]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entry in entries:
        key = normalize_key(entry.term)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry.term)
    return out


def write_jsonl_entries(path: Path, entries: Sequence[TermEntry]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in documents_for_entries(entries):
            f.write(json.dumps({"text": doc.text}, ensure_ascii=False))
            f.write("\n")


def format_entries(entries: Sequence[TermEntry]) -> str:
    text = "\n\n".join(doc.text for doc in documents_for_entries(entries))
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def write_text_split(
    out_dir: Path, train_entries: Sequence[TermEntry], val_entries: Sequence[TermEntry]
) -> CorpusSplit:
    raw_dir = out_dir / "raw_ita_dict"
    ensure_clean_dir(raw_dir)
    train_path = raw_dir / "train.txt"
    val_path = raw_dir / "val.txt"
    train_jsonl_path = raw_dir / "train.jsonl"
    val_jsonl_path = raw_dir / "val.jsonl"
    train_text = format_entries(train_entries)
    val_text = format_entries(val_entries)
    train_documents = len(documents_for_entries(train_entries))
    val_documents = len(documents_for_entries(val_entries))
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    write_jsonl_entries(train_jsonl_path, train_entries)
    write_jsonl_entries(val_jsonl_path, val_entries)
    return CorpusSplit(
        train_terms=unique_terms(train_entries),
        val_terms=unique_terms(val_entries),
        train_chars=len(train_text),
        val_chars=len(val_text),
        train_documents=train_documents,
        val_documents=val_documents,
        train_sha256=file_sha256(train_path),
        val_sha256=file_sha256(val_path) if val_text else "",
        train_jsonl_sha256=file_sha256(train_jsonl_path),
        val_jsonl_sha256=file_sha256(val_jsonl_path) if val_entries else "",
    )


def prepare_tokenized_corpus(
    *,
    split_name: str,
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    vocab_size: int,
    min_frequency: int,
    use_existing_tokenizer: bool,
) -> TokenizedCorpus:
    ensure_clean_dir(output_dir)
    summary = tokenize_jsonl(
        tokenizer=tokenizer_path,
        input_jsonl=input_path,
        output_dir=output_dir,
        im_start=IM_START,
        im_end=IM_END,
        max_length=2_147_483_647,
        jsonl_text_field="text",
        use_tokenizer=use_existing_tokenizer,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=(PAD,),
        seed=17,
    )
    action = (
        "tokenized with existing tokenizer"
        if use_existing_tokenizer
        else "trained tokenizer"
    )
    print(
        f"{split_name}: {action}: {summary['tokenizer']} "
        f"(source_documents={summary['sequences']}, tokens_with_delimiters={summary['tokens']})"
    )
    return TokenizedCorpus.open(output_dir)


def i32_tensor(array: np.ndarray) -> TensorData:
    arr = np.asarray(array, dtype=TOKEN_DTYPE)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return TensorData(dtype="i32", shape=(int(arr.size),), data=arr.tobytes(order="C"))


def u16_tensor(array: np.ndarray) -> TensorData:
    arr_i64 = np.asarray(array, dtype=np.int64)
    if arr_i64.size == 0:
        raise ValueError("token rows cannot be empty")
    if int(arr_i64.min()) < 0 or int(arr_i64.max()) > 0xFFFF:
        raise ValueError("token id does not fit in uint16 compact storage")
    arr = arr_i64.astype(np.dtype("<u2"), copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return TensorData(dtype="u16", shape=(int(arr.size),), data=arr.tobytes(order="C"))


def _segment_lengths_and_positions(
    doc_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention segment lengths and per-segment positions for one row."""

    if doc_ids.ndim != 1 or doc_ids.size == 0:
        raise ValueError("doc_ids must be a non-empty vector")
    change = np.flatnonzero(doc_ids[1:] != doc_ids[:-1]) + 1
    bounds = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            change.astype(np.int64, copy=False),
            np.asarray([doc_ids.size], dtype=np.int64),
        ]
    )
    lengths = np.diff(bounds).astype(TOKEN_DTYPE, copy=False)
    positions = np.empty(doc_ids.size, dtype=TOKEN_DTYPE)
    for begin, end in zip(bounds[:-1], bounds[1:]):
        positions[begin:end] = np.arange(end - begin, dtype=TOKEN_DTYPE)
    return lengths, positions


def iter_compact_lm_samples(corpus: TokenizedCorpus, context_length: int):
    """Yield compact fixed-length LM rows plus document-fragment metadata."""

    record_length = context_length + 1
    if corpus.mode != "jsonl":
        raise ValueError(f"expected jsonl tokenized corpus, got {corpus.mode!r}")
    if record_length < 3:
        raise ValueError("record_length must be >= 3")

    parts = [np.asarray(seq, dtype=TOKEN_DTYPE) for seq in corpus.iter_sequences()]
    if not parts:
        return
    stream = np.concatenate(parts)
    doc_ids = np.concatenate(
        [np.full(part.size, i, dtype=np.int32) for i, part in enumerate(parts)]
    )
    if stream.size != doc_ids.size:
        raise ValueError("token/doc-id stream size mismatch")
    for start in range(0, stream.size - record_length + 1, context_length):
        row = stream[start : start + record_length]
        row_doc_ids = doc_ids[start : start + record_length]
        segment_lengths, _positions = _segment_lengths_and_positions(row_doc_ids[:-1])
        if int(segment_lengths.sum()) != context_length:
            raise ValueError("attention segment lengths do not cover the fixed row")
        yield {
            "tokens": u16_tensor(row),
            "segment_lengths": i32_tensor(segment_lengths),
        }


def write_gdds_dataset(
    *,
    out_dir: Path,
    train_corpus: TokenizedCorpus,
    val_corpus: TokenizedCorpus | None,
    context_length: int,
    max_shard_bytes: int,
) -> dict[str, list[Path]]:
    if train_corpus.mode != "jsonl":
        raise ValueError(f"expected jsonl training corpus, got {train_corpus.mode!r}")
    if val_corpus is not None and val_corpus.mode != "jsonl":
        raise ValueError(f"expected jsonl validation corpus, got {val_corpus.mode!r}")
    fields = storage_fields_for_context(context_length)
    remove_split_shards(out_dir, "train")
    remove_split_shards(out_dir, "val")
    train_writer = GddsSplitWriter(
        out_dir, "train", fields, max_shard_bytes=max_shard_bytes
    )
    val_writer = GddsSplitWriter(
        out_dir, "val", fields, max_shard_bytes=max_shard_bytes
    )
    try:
        for sample in iter_compact_lm_samples(train_corpus, context_length):
            train_writer.write_sample(sample)
        if val_corpus is not None:
            for sample in iter_compact_lm_samples(val_corpus, context_length):
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
    source_dir: Path,
    source_glob: str,
    tokenizer_path: Path,
    split: CorpusSplit,
    source_stats: Mapping[str, object],
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
    storage_fields = storage_fields_for_context(context_length)
    runtime_fields = fields_for_context(context_length)
    pad_id = tokenizer_special_id(tokenizer_path, PAD)
    split_samples = {
        split_name: sum(read_gdds_header(path).samples for path in paths)
        for split_name, paths in gdds_paths.items()
    }
    val_records = 0 if val_corpus is None else val_corpus.num_sequences
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    manifest = {
        "format": "GDDS",
        "version": 1,
        "schema_hash": f"0x{schema_hash(storage_fields):016x}",
        "fields": [field_metadata(field_spec) for field_spec in storage_fields],
        "runtime_schema_hash": f"0x{schema_hash(runtime_fields):016x}",
        "runtime_fields": [field_metadata(field_spec) for field_spec in runtime_fields],
        "storage": {
            "tokens": f"per-sample uint16 token ids, fixed length {context_length + 1}; runtime transform emits shifted i32 inputs/targets",
            "segment_lengths": "per-row document-fragment lengths over tokens[:-1], summing to context_length",
            "input_ids": "runtime transform: tokens[:-1] cast to i32; batch-collated packed",
            "positions": "runtime transform: per-token positions reset at each document fragment",
            "target_ids": "runtime transform: tokens[1:] cast to i32; cross-document targets replaced by pad_id and ignored by LM CE",
            "cu_seqlens": "generated by GDDS dataloader from runtime segment_lengths; shape [sum document fragments + 1]",
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "hash": train_corpus.manifest["tokenizer_hash"],
            "vocab_size": vocab_size,
            "pad_id": pad_id,
            "im_start_id": train_corpus.im_start_id,
            "im_end_id": train_corpus.im_end_id,
            "digits": "always split before BPE merges",
            "trained_on": "raw_ita_dict/train.jsonl",
            "train_sha256": split.train_jsonl_sha256,
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
            "mode": "term_random_before_tokenizer_training",
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
            "train_path": "raw_ita_dict/train.txt",
            "val_path": "raw_ita_dict/val.txt",
            "train_jsonl_sha256": split.train_jsonl_sha256,
            "val_jsonl_sha256": split.val_jsonl_sha256,
            "train_jsonl_path": "raw_ita_dict/train.jsonl",
            "val_jsonl_path": "raw_ita_dict/val.jsonl",
        },
        "dictionary_format": {
            "source_record": "one Markdown wiki file per source term, optionally paired with .story.md and .quiz.md siblings",
            "output_record": "one JSONL row per document with a `text` field; term, story, and quiz are separate documents",
            "raw_text_entry_separator": "blank line in raw_ita_dict/*.txt only",
            "entry_header": None,
            "term_line": "Termine: <lemma>",
            "term_document": {
                "body": "the original <term>.md Markdown structure is preserved except BOM/CRLF normalization and removal of the leading '# ' before 'Termine:'",
            },
            "story_document": {
                "header": "## Story",
                "body": "normalized text from the .story.md sibling, preserving paragraph breaks",
            },
            "quiz_document": {
                "header": "## Quiz",
                "body": "one normalized Q&A block from the .quiz.md sibling; blocks are split on Markdown horizontal rules (`---`)",
            },
            "entry_footer": None,
            "sequence_delimiters": "tokenizer inserts <|im_start|>/<|im_end|> around every JSONL row before stream packing",
            "preprocessing": [
                "ignore .story.md and .quiz.md files as standalone term pages and load them only as sibling documents",
                "parse term Markdown only for metadata/statistics and term-level splitting",
                "emit each <term>.md as its own document preserving the source Markdown structure except the leading '# ' term heading marker",
                "normalize story/quiz line endings and trailing whitespace while preserving paragraph breaks",
                "split quiz siblings on `---`, producing one JSONL document per Q&A block",
                "keep duplicate source term files as separate term documents",
                "split train/val by normalized term so duplicate terms do not cross splits",
            ],
            "max_examples_per_definition": max_examples_per_definition,
        },
        "source": {
            "source_dir": str(source_dir),
            "source_glob": source_glob,
            **dict(source_stats),
        },
        "prep": {
            "format_version": DATA_FORMAT_VERSION,
            "max_shard_bytes": max_shard_bytes,
            "attention_boundaries": "document-local segments inside each fixed row via stored segment_lengths",
            "ignore_index": pad_id,
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


def read_gdds_lm_sample(shards: Sequence[Path], sample_index: int) -> np.ndarray:
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
            payload_offset, payload_nbytes = struct.unpack_from(
                "<QQ", record, base + 72
            )
            data = record[
                header_nbytes + payload_offset : header_nbytes
                + payload_offset
                + payload_nbytes
            ]
            dtype = np.dtype("<u2") if field_id == 0 else TOKEN_DTYPE
            arrays[field_id] = np.frombuffer(data, dtype=dtype).reshape(dims).copy()
        if 0 not in arrays:
            raise ValueError(f"missing compact tokens in {path}")
        return arrays[0].astype(TOKEN_DTYPE, copy=False)
    raise IndexError(sample_index)


def show_random_samples(
    shards: Sequence[Path],
    tokenizer_path: Path,
    total_samples: int,
    count: int,
    seed: int,
) -> None:
    if count <= 0 or total_samples <= 0:
        return
    decoder = TokenDecoder(tokenizer_path)
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_samples), k=min(count, total_samples)))
    print(f"\nDecoded random train samples ({len(selected)}):")
    for sample_index in selected:
        full = read_gdds_lm_sample(shards, sample_index)
        text = decoder.decode(full)
        print(f"\n--- sample {sample_index} ({len(full)} tokens) ---")
        print(text[:900].replace("\0", "�"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--story-suffix", default=DEFAULT_STORY_SUFFIX)
    parser.add_argument("--quiz-suffix", default=DEFAULT_QUIZ_SUFFIX)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--max-examples-per-definition",
        type=int,
        default=DEFAULT_MAX_EXAMPLES_PER_DEFINITION,
        help="use -1 to keep all examples",
    )
    parser.add_argument(
        "--keep-none-fields",
        action="store_true",
        help="keep exact `nessuno` synonym/antonym placeholders instead of dropping them",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--preview-seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if args.vocab_size < 259:
        raise ValueError(
            "--vocab-size must be at least 259 for byte tokens plus pad/start/end specials"
        )
    if args.min_frequency <= 0:
        raise ValueError("--min-frequency must be positive")
    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1")
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")

    entries, stats = load_entries(
        source_dir=source_dir,
        source_glob=args.source_glob,
        max_examples_per_definition=args.max_examples_per_definition,
        drop_none_fields=not args.keep_none_fields,
        story_suffix=args.story_suffix,
        quiz_suffix=args.quiz_suffix,
    )
    if not entries:
        raise ValueError("no dictionary entries loaded")
    print(
        "Loaded Markdown dictionary: "
        f"source_files={stats['source_files']} parsed_files={stats['parsed_files']} "
        f"terms={stats['terms']} documents={stats.get('documents', 0)} "
        f"definitions={stats['definitions']} examples={stats['examples']} "
        f"stories={stats.get('merged_stories', 0)} quizzes={stats.get('merged_quizzes', 0)}"
    )
    print(
        "Preprocessing: "
        f"duplicate_term_groups={stats['duplicate_term_groups']} "
        f"non_sequential_numbering_files={stats.get('non_sequential_numbering_files', 0)} "
        f"story_files={stats.get('story_files', 0)} "
        f"missing_story_files={stats.get('missing_story_files', 0)} "
        f"quiz_files={stats.get('quiz_files', 0)} "
        f"missing_quiz_files={stats.get('missing_quiz_files', 0)} "
        f"dropped_none_synonyms={stats.get('dropped_none_Sinonimi', 0)} "
        f"dropped_none_antonyms={stats.get('dropped_none_Contrari', 0)}"
    )

    train_entries, val_entries = split_entries(
        entries, args.val_fraction, args.split_seed
    )
    ensure_clean_dir(out_dir)
    tokenizer_path = out_dir / f"tokenizer-v{args.vocab_size}.json"
    split = write_text_split(out_dir, train_entries, val_entries)
    print(
        "Raw term split: "
        f"train_terms={len(split.train_terms)} val_terms={len(split.val_terms)} "
        f"train_documents={split.train_documents} val_documents={split.val_documents} "
        f"train_chars={split.train_chars} val_chars={split.val_chars} "
        f"val_fraction={args.val_fraction:.3f} seed={args.split_seed}"
    )

    train_corpus = prepare_tokenized_corpus(
        split_name="train",
        input_path=out_dir / "raw_ita_dict" / "train.jsonl",
        output_dir=out_dir / "tokenized_train",
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        use_existing_tokenizer=False,
    )
    val_corpus = None
    if val_entries:
        val_corpus = prepare_tokenized_corpus(
            split_name="val",
            input_path=out_dir / "raw_ita_dict" / "val.jsonl",
            output_dir=out_dir / "tokenized_val",
            tokenizer_path=tokenizer_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            use_existing_tokenizer=True,
        )

    if train_corpus.mode != "jsonl":
        raise ValueError(f"unexpected train tokenized mode {train_corpus.mode!r}")
    if val_corpus is not None and val_corpus.mode != "jsonl":
        raise ValueError(f"unexpected val tokenized mode {val_corpus.mode!r}")

    paths = write_gdds_dataset(
        out_dir=out_dir,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        context_length=args.context_length,
        max_shard_bytes=args.max_shard_bytes,
    )
    write_manifest(
        out_dir=out_dir,
        source_dir=source_dir,
        source_glob=args.source_glob,
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

    total_samples = sum(
        read_gdds_header(path).samples
        for split_paths in paths.values()
        for path in split_paths
    )
    train_samples = sum(read_gdds_header(path).samples for path in paths["train"])
    val_samples = sum(read_gdds_header(path).samples for path in paths.get("val", []))
    val_tokens = 0 if val_corpus is None else val_corpus.num_tokens
    print(
        f"Wrote {total_samples} GPT LM samples to {out_dir} "
        f"(train={train_samples}, val={val_samples})"
    )
    print(
        "Token counts with delimiters: "
        f"train={train_corpus.num_tokens} val={val_tokens} "
        f"total={train_corpus.num_tokens + val_tokens}"
    )
    print(
        "Packed training tokens: "
        f"train={train_samples * args.context_length} "
        f"val={val_samples * args.context_length}"
    )
    for split_paths in paths.values():
        for path in split_paths:
            print(path)
    print(out_dir / "manifest.json")

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
