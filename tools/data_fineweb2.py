#!/usr/bin/env python3
# /// script
# dependencies = ["datasets>=2.18.0", "numpy>=1.26", "python-dotenv>=1.0.0"]
# ///
"""Build a FineWeb/FineWeb2 GPT-LM GDDS dataset.

Current implementation status: raw data collection plus shared tokenizer corpus
building/training are implemented; later stages are intentionally scaffolded
with ``NotImplementedError`` while we settle the script structure.

Target corpus:
  * English from ``HuggingFaceFW/fineweb``.
  * Top 9 non-English language-script subsets from ``HuggingFaceFW/fineweb-2``.
  * One shared byte-BPE tokenizer.
  * Packed 2048-token GPT-LM samples in GDDS, using compact u16 token storage
    plus per-row document-fragment ``segment_lengths`` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

PAD = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_OUT_ROOT = Path("/Volumes/Seagate 2TB/datasets/gd_fineweb2")
DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_TOTAL_PACKED_TOKENS = 1_000_000_000
DEFAULT_VAL_PACKED_TOKENS = 10_000_000
DEFAULT_VOCAB_SIZE = 65_536
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_TOKENIZER_TRAIN_EST_TOKENS = 80_000_000
DEFAULT_RAW_OVERFETCH = 1.08
DEFAULT_VAL_FRACTION = DEFAULT_VAL_PACKED_TOKENS / DEFAULT_TOTAL_PACKED_TOKENS
DEFAULT_SEED = 17
DEFAULT_SHUFFLE_BUFFER = 1_000
DEFAULT_WORKERS = 4
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 0  # 0 means unbounded
MAX_RAW_SHARD_BYTES = 512 * 1024 * 1024
DEFAULT_RAW_WRITE_BUFFER_BYTES = 8 * 1024 * 1024
MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
DATA_FORMAT_VERSION = "gpt-lm-fineweb2-plus-fineweb-en-v1-2048-u16-segments"


class Stage(str, Enum):
    """Named resumable pipeline stages."""

    COLLECT_RAW = "collect-raw"
    BUILD_TOKENIZER_CORPUS = "build-tokenizer-corpus"
    TRAIN_TOKENIZER = "train-tokenizer"
    TOKENIZE = "tokenize"
    PACK = "pack"
    VERIFY = "verify"
    WRITE_MANIFEST = "write-manifest"


@dataclass(frozen=True)
class SourceSpec:
    """A streamed source subset and its normalized language key."""

    language: str
    dataset: str
    config: str | None
    split: str = "train"
    text_field: str = "text"
    id_fields: tuple[str, ...] = ("id", "url", "dump")


DEFAULT_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("eng_Latn", "HuggingFaceFW/fineweb", "sample-10BT"),
    SourceSpec("rus_Cyrl", "HuggingFaceFW/fineweb-2", "rus_Cyrl"),
    SourceSpec("cmn_Hani", "HuggingFaceFW/fineweb-2", "cmn_Hani"),
    SourceSpec("jpn_Jpan", "HuggingFaceFW/fineweb-2", "jpn_Jpan"),
    SourceSpec("deu_Latn", "HuggingFaceFW/fineweb-2", "deu_Latn"),
    SourceSpec("spa_Latn", "HuggingFaceFW/fineweb-2", "spa_Latn"),
    SourceSpec("fra_Latn", "HuggingFaceFW/fineweb-2", "fra_Latn"),
    SourceSpec("ita_Latn", "HuggingFaceFW/fineweb-2", "ita_Latn"),
    SourceSpec("por_Latn", "HuggingFaceFW/fineweb-2", "por_Latn"),
    SourceSpec("nld_Latn", "HuggingFaceFW/fineweb-2", "nld_Latn"),
)


@dataclass(frozen=True)
class PipelinePaths:
    """Canonical output locations for the dataset build."""

    out_root: Path
    raw_dir: Path
    tokenizer_dir: Path
    tokenizer_train_path: Path
    tokenizer_path: Path
    tokenized_dir: Path
    gdds_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Complete immutable configuration for a build run."""

    out_root: Path = DEFAULT_OUT_ROOT
    num_samples: int = 0
    stages: tuple[Stage, ...] = (Stage.COLLECT_RAW,)
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCE_SPECS
    context_length: int = DEFAULT_CONTEXT_LENGTH
    total_packed_tokens: int = DEFAULT_TOTAL_PACKED_TOKENS
    val_packed_tokens: int = DEFAULT_VAL_PACKED_TOKENS
    vocab_size: int = DEFAULT_VOCAB_SIZE
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    tokenizer_train_est_tokens: int = DEFAULT_TOKENIZER_TRAIN_EST_TOKENS
    raw_overfetch: float = DEFAULT_RAW_OVERFETCH
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER
    workers: int = DEFAULT_WORKERS
    min_chars: int = DEFAULT_MIN_CHARS
    max_chars: int = DEFAULT_MAX_CHARS
    raw_shard_bytes: int = MAX_RAW_SHARD_BYTES
    raw_write_buffer_bytes: int = DEFAULT_RAW_WRITE_BUFFER_BYTES
    max_shard_bytes: int = MAX_SHARD_BYTES
    force: bool = False
    resume: bool = True


@dataclass(frozen=True)
class LanguageBudget:
    """Per-language raw, tokenization, and final packed-sample targets."""

    source: SourceSpec
    raw_documents: int
    train_packed_samples: int
    val_packed_samples: int
    train_packed_tokens: int
    val_packed_tokens: int
    raw_estimated_tokens: int
    tokenizer_train_estimated_tokens: int


@dataclass(frozen=True)
class RawDocument:
    """Normalized document record persisted to raw JSONL shards."""

    text: str
    language: str
    split: str
    source_dataset: str
    source_config: str | None
    source_split: str
    record_id: str
    url: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RawShardRef:
    """Raw JSONL shard produced by the collection stage."""

    path: Path
    language: str
    split: str
    documents: int
    estimated_tokens: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TokenizerRef:
    """Saved tokenizer metadata used by later stages."""

    path: Path
    vocab_size: int
    hash: int
    pad_id: int
    im_start_id: int
    im_end_id: int


@dataclass(frozen=True)
class TokenizedShardRef:
    """Tokenized JSONL shard directory produced by gradients-tokenize."""

    path: Path
    source_raw_shard: Path
    language: str
    split: str
    documents: int
    tokens: int


@dataclass(frozen=True)
class PackedShardRef:
    """Final GDDS shard metadata."""

    path: Path
    language: str
    split: str
    samples: int
    packed_tokens: int
    schema_hash: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class StageSummary:
    """Generic stage result included in the final manifest."""

    stage: Stage
    status: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PipelineSummary:
    """Top-level result returned by ``run_pipeline``."""

    config: PipelineConfig
    paths: PipelinePaths
    budgets: tuple[LanguageBudget, ...]
    stages: tuple[StageSummary, ...]
    manifest_path: Path


@dataclass(frozen=True)
class PackedSample:
    """One compact LM sample before GDDS serialization."""

    tokens: object
    segment_lengths: object
    language: str
    split: str


# ---------------------------------------------------------------------------
# CLI / orchestration entrypoints
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    load_project_dotenv()
    args = parse_args(argv)
    config = config_from_args(args)
    summary = run_pipeline(config)
    print(json.dumps(to_jsonable(pipeline_cli_summary(summary)), indent=2, ensure_ascii=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments into an argparse namespace."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="output root; defaults to the external dataset disk",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=parse_count,
        help="total raw documents to collect across all selected sources, e.g. 1000 or 1k",
    )
    parser.add_argument(
        "--stages",
        default=Stage.COLLECT_RAW.value,
        help="comma-separated stages; only collect-raw is implemented today",
    )
    parser.add_argument(
        "--languages",
        default="",
        help="comma-separated language keys to collect; default is English plus top 9 FineWeb2 languages",
    )
    parser.add_argument(
        "--english-config",
        default="sample-10BT",
        help="FineWeb config for eng_Latn",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER,
        help="HF streaming shuffle buffer; default is small for smoke tests, raise for full runs",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel source collectors")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--raw-shard-bytes", type=parse_count, default=MAX_RAW_SHARD_BYTES)
    parser.add_argument(
        "--raw-write-buffer-bytes",
        type=parse_count,
        default=DEFAULT_RAW_WRITE_BUFFER_BYTES,
        help="per-shard buffered write size; useful for HDD targets",
    )
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--total-packed-tokens", type=parse_count, default=DEFAULT_TOTAL_PACKED_TOKENS)
    parser.add_argument("--val-packed-tokens", type=parse_count, default=DEFAULT_VAL_PACKED_TOKENS)
    parser.add_argument("--tokenizer-train-est-tokens", type=parse_count, default=DEFAULT_TOKENIZER_TRAIN_EST_TOKENS)
    parser.add_argument("--raw-overfetch", type=float, default=DEFAULT_RAW_OVERFETCH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument("--max-shard-bytes", type=parse_count, default=MAX_SHARD_BYTES)
    parser.add_argument("--force", action="store_true", help="remove existing raw collection outputs before collecting")
    parser.add_argument("--no-resume", action="store_true", help="fail instead of reusing an existing raw manifest")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed CLI arguments into ``PipelineConfig``."""

    selected_languages = parse_csv(args.languages)
    stages = tuple(parse_stage_list(args.stages))
    sources = []
    for source in DEFAULT_SOURCE_SPECS:
        if selected_languages and source.language not in selected_languages:
            continue
        if source.language == "eng_Latn":
            source = replace(source, config=args.english_config)
        sources.append(source)
    return PipelineConfig(
        out_root=args.out_dir.expanduser(),
        num_samples=args.num_samples,
        stages=stages,
        sources=tuple(sources),
        context_length=args.context_length,
        total_packed_tokens=args.total_packed_tokens,
        val_packed_tokens=args.val_packed_tokens,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        tokenizer_train_est_tokens=args.tokenizer_train_est_tokens,
        raw_overfetch=args.raw_overfetch,
        val_fraction=args.val_fraction,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        workers=args.workers,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        raw_shard_bytes=args.raw_shard_bytes,
        raw_write_buffer_bytes=args.raw_write_buffer_bytes,
        max_shard_bytes=args.max_shard_bytes,
        force=args.force,
        resume=not args.no_resume,
    )


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    """Run selected stages in dependency order."""

    validate_config(config)
    paths = resolve_paths(config)
    budgets = compute_language_budgets(config)
    ensure_output_layout(paths, force=config.force and Stage.COLLECT_RAW in config.stages, resume=config.resume)

    summaries: list[StageSummary] = []
    for stage in config.stages:
        if stage == Stage.COLLECT_RAW:
            summaries.append(collect_raw_stage(config, paths, budgets))
        elif stage == Stage.BUILD_TOKENIZER_CORPUS:
            summaries.append(build_tokenizer_corpus_stage(config, paths, budgets))
        elif stage == Stage.TRAIN_TOKENIZER:
            summaries.append(train_tokenizer_stage(config, paths))
        elif stage == Stage.TOKENIZE:
            summaries.append(tokenize_stage(config, paths, load_tokenizer_ref(paths.tokenizer_path)))
        elif stage == Stage.PACK:
            summaries.append(pack_stage(config, paths, budgets, load_tokenizer_ref(paths.tokenizer_path)))
        elif stage == Stage.VERIFY:
            summaries.append(verify_stage(config, paths, load_tokenizer_ref(paths.tokenizer_path)))
        elif stage == Stage.WRITE_MANIFEST:
            summaries.append(write_manifest_stage(config, paths, summaries))
        else:
            raise ValueError(f"unsupported stage {stage!r}")

    manifest_path = summaries[-1].artifacts[0] if summaries and summaries[-1].artifacts else paths.manifest_path
    return PipelineSummary(
        config=config,
        paths=paths,
        budgets=budgets,
        stages=tuple(summaries),
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# Planning, budgets, and paths
# ---------------------------------------------------------------------------


def resolve_paths(config: PipelineConfig) -> PipelinePaths:
    """Resolve all output paths from the root config."""

    out_root = config.out_root
    tokenizer_dir = out_root / "tokenizer"
    return PipelinePaths(
        out_root=out_root,
        raw_dir=out_root / "raw",
        tokenizer_dir=tokenizer_dir,
        tokenizer_train_path=tokenizer_dir / "tokenizer-train.jsonl",
        tokenizer_path=tokenizer_dir / f"tokenizer-v{config.vocab_size}.json",
        tokenized_dir=out_root / "tokenized",
        gdds_dir=out_root / "gdds",
        manifest_path=out_root / "manifest.json",
    )


def validate_config(config: PipelineConfig) -> None:
    """Validate context length, vocab size, source list, stages, and budgets."""

    if config.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not config.sources:
        raise ValueError("at least one source/language is required")
    if len({source.language for source in config.sources}) != len(config.sources):
        raise ValueError("source language keys must be unique")
    if not config.stages:
        raise ValueError("at least one stage is required")
    if config.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if config.vocab_size < 259 or config.vocab_size > 65_536:
        raise ValueError("--vocab-size must be in [259, 65536] for compact u16 storage")
    if config.min_frequency <= 0:
        raise ValueError("--min-frequency must be positive")
    if config.shuffle_buffer < 0:
        raise ValueError("--shuffle-buffer must be non-negative")
    if config.workers <= 0:
        raise ValueError("--workers must be positive")
    if not (0.0 <= config.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 <= val_fraction < 1")
    if config.min_chars < 0 or config.max_chars < 0:
        raise ValueError("--min-chars/--max-chars must be non-negative")
    if config.max_chars and config.max_chars < config.min_chars:
        raise ValueError("--max-chars must be >= --min-chars when set")
    if config.raw_shard_bytes <= 0:
        raise ValueError("--raw-shard-bytes must be positive")
    if config.raw_write_buffer_bytes <= 0:
        raise ValueError("--raw-write-buffer-bytes must be positive")
    if config.total_packed_tokens <= 0 or config.val_packed_tokens < 0:
        raise ValueError("packed token targets must be valid")
    if config.val_packed_tokens >= config.total_packed_tokens:
        raise ValueError("--val-packed-tokens must be smaller than --total-packed-tokens")


def compute_language_budgets(config: PipelineConfig) -> tuple[LanguageBudget, ...]:
    """Allocate train/val packed-sample targets and raw/tokenizer collection budgets."""

    n_sources = len(config.sources)
    raw_documents = allocate_evenly(config.num_samples, n_sources)
    total_samples = math.ceil(config.total_packed_tokens / config.context_length)
    val_samples = math.ceil(config.val_packed_tokens / config.context_length)
    train_samples = max(0, total_samples - val_samples)
    train_samples_by_lang = allocate_evenly(train_samples, n_sources)
    val_samples_by_lang = allocate_evenly(val_samples, n_sources)
    tokenizer_tokens_by_lang = allocate_evenly(config.tokenizer_train_est_tokens, n_sources)

    budgets = []
    for i, source in enumerate(config.sources):
        train_tokens = train_samples_by_lang[i] * config.context_length
        val_tokens = val_samples_by_lang[i] * config.context_length
        budgets.append(
            LanguageBudget(
                source=source,
                raw_documents=raw_documents[i],
                train_packed_samples=train_samples_by_lang[i],
                val_packed_samples=val_samples_by_lang[i],
                train_packed_tokens=train_tokens,
                val_packed_tokens=val_tokens,
                raw_estimated_tokens=math.ceil((train_tokens + val_tokens) * config.raw_overfetch),
                tokenizer_train_estimated_tokens=tokenizer_tokens_by_lang[i],
            )
        )
    return tuple(budgets)


def allocate_evenly(total: int, buckets: int) -> tuple[int, ...]:
    """Allocate an integer total across buckets with deterministic remainder handling."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    base, remainder = divmod(total, buckets)
    return tuple(base + (1 if i < remainder else 0) for i in range(buckets))


def ensure_output_layout(paths: PipelinePaths, *, force: bool, resume: bool) -> None:
    """Create or clean the canonical output directory layout."""

    if force and paths.raw_dir.exists():
        # Some external filesystems on macOS expose transient AppleDouble
        # ``._*`` sidecar files; ignore disappear-during-delete races.
        shutil.rmtree(paths.raw_dir, ignore_errors=True)
    paths.out_root.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    paths.tokenized_dir.mkdir(parents=True, exist_ok=True)
    paths.gdds_dir.mkdir(parents=True, exist_ok=True)
    if not resume and not force and (paths.raw_dir / "manifest.json").exists():
        raise FileExistsError(paths.raw_dir / "manifest.json")


# ---------------------------------------------------------------------------
# Stage: collect raw documents
# ---------------------------------------------------------------------------


def collect_raw_stage(config: PipelineConfig, paths: PipelinePaths, budgets: Sequence[LanguageBudget]) -> StageSummary:
    """Stream FineWeb/FineWeb2, split deterministically, and write raw JSONL shards.

    Resume is per-language.  A language is considered complete only after its
    ``raw/<language>/manifest.json`` is atomically written.  If a previous run
    died mid-language, the partial language directory is deleted and that
    language is replayed from the deterministic HF stream; already completed
    languages are reused.
    """

    manifest_path = paths.raw_dir / "manifest.json"
    if manifest_path.exists() and not config.force and not config.resume:
        raise FileExistsError(manifest_path)

    max_workers = min(config.workers, len(budgets))
    print(
        f"collect-raw: languages={len(budgets)} workers={max_workers} "
        f"shuffle_buffer={config.shuffle_buffer} text_only=true resume={config.resume}",
        flush=True,
    )

    results: list[tuple[int, tuple[RawShardRef, ...], dict[str, object]]] = []
    pending: list[tuple[int, LanguageBudget]] = []
    reused_languages = 0
    restarted_languages = 0

    for source_index, budget in enumerate(budgets):
        existing = load_completed_raw_language(config, paths, budget, source_index)
        if existing is not None and config.resume and not config.force:
            shards, metrics = existing
            metrics = {**metrics, "resume_status": "reused"}
            results.append((source_index, shards, metrics))
            reused_languages += 1
            print(f"collect-raw[{budget.source.language}]: resume reuse complete language", flush=True)
            continue
        language_dir = raw_language_dir(paths, budget.source.language)
        if language_dir.exists():
            if config.force:
                shutil.rmtree(language_dir, ignore_errors=True)
                restarted_languages += 1
                print(f"collect-raw[{budget.source.language}]: removed raw dir due to --force", flush=True)
            elif config.resume:
                if raw_language_dir_compatible_for_resume(config, paths, budget, source_index):
                    print(f"collect-raw[{budget.source.language}]: resume partial language", flush=True)
                else:
                    shutil.rmtree(language_dir, ignore_errors=True)
                    restarted_languages += 1
                    print(f"collect-raw[{budget.source.language}]: removed incompatible raw dir", flush=True)
            else:
                raise FileExistsError(language_dir)
        pending.append((source_index, budget))

    if pending:
        worker_count = min(max_workers, len(pending))
        results.extend(collect_raw_languages_multiprocess(config, paths, pending, worker_count))

    results.sort(key=lambda item: item[0])
    all_shards = [shard for _idx, shards, _metrics in results for shard in shards]
    source_metrics = [metrics for _idx, _shards, metrics in results]
    metrics = {
        "documents": sum(shard.documents for shard in all_shards),
        "estimated_tokens": sum(shard.estimated_tokens for shard in all_shards),
        "bytes": sum(shard.bytes for shard in all_shards),
        "shards": len(all_shards),
        "languages": len(budgets),
        "workers": max_workers,
        "reused_languages": reused_languages,
        "collected_languages": len(pending),
        "restarted_languages": restarted_languages,
        "shuffle_requested": config.shuffle_buffer > 0,
        "shuffle_buffer": config.shuffle_buffer,
        "raw_schema": "text-only-jsonl",
        "write_buffer_bytes": config.raw_write_buffer_bytes,
        "sources": source_metrics,
    }
    manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "stage": Stage.COLLECT_RAW.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": to_jsonable(config),
        "paths": to_jsonable(paths),
        "budgets": to_jsonable(budgets),
        "metrics": metrics,
        "shards": [to_jsonable(shard) for shard in all_shards],
    }
    write_json_atomic(manifest_path, manifest)
    status = "skipped-existing" if not pending else ("resumed" if reused_languages else "ok")
    return StageSummary(
        stage=Stage.COLLECT_RAW,
        status=status,
        metrics=metrics,
        artifacts=tuple([manifest_path, *[shard.path for shard in all_shards]]),
    )


def raw_language_dir(paths: PipelinePaths, language: str) -> Path:
    """Return the raw directory for one language."""

    return paths.raw_dir / language


def raw_language_manifest_path(paths: PipelinePaths, language: str) -> Path:
    """Return the per-language raw completion manifest path."""

    return raw_language_dir(paths, language) / "manifest.json"


def raw_language_resume_key(config: PipelineConfig, budget: LanguageBudget, source_index: int) -> Mapping[str, object]:
    """Return the config subset that makes a raw language shard set reusable."""

    return {
        "format_version": DATA_FORMAT_VERSION,
        "raw_schema": "text-only-jsonl",
        "language": budget.source.language,
        "source": to_jsonable(budget.source),
        "source_index": source_index,
        "target_documents": budget.raw_documents,
        "seed": config.seed,
        "effective_seed": config.seed + source_index,
        "shuffle_buffer": config.shuffle_buffer,
        "val_fraction": config.val_fraction,
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
        "raw_shard_bytes": config.raw_shard_bytes,
    }


def load_completed_raw_language(
    config: PipelineConfig,
    paths: PipelinePaths,
    budget: LanguageBudget,
    source_index: int,
) -> tuple[tuple[RawShardRef, ...], dict[str, object]] | None:
    """Load a completed per-language raw manifest if it matches this run."""

    manifest_path = raw_language_manifest_path(paths, budget.source.language)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("stage") != Stage.COLLECT_RAW.value or manifest.get("status") != "ok":
        return None
    if manifest.get("resume_key") != raw_language_resume_key(config, budget, source_index):
        return None
    shards = tuple(raw_shard_from_manifest(row) for row in manifest.get("shards", []))
    if not shards:
        return None
    if sum(shard.documents for shard in shards) != budget.raw_documents:
        return None
    for shard in shards:
        if not shard.path.exists():
            return None
        if shard.path.stat().st_size != shard.bytes:
            return None
    metrics = dict(manifest.get("metrics", {}))
    if int(metrics.get("kept_documents", 0)) != budget.raw_documents:
        return None
    return shards, metrics


def raw_language_dir_compatible_for_resume(
    config: PipelineConfig,
    paths: PipelinePaths,
    budget: LanguageBudget,
    source_index: int,
) -> bool:
    """Return whether an existing language dir can be incrementally resumed."""

    manifest_path = raw_language_manifest_path(paths, budget.source.language)
    if not manifest_path.exists():
        # Legacy/partial dir from before progress manifests.  We can still scan
        # text-only JSONL shards and replay the deterministic stream prefix.
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("stage") != Stage.COLLECT_RAW.value:
        return False
    return manifest.get("resume_key") == raw_language_resume_key(config, budget, source_index)


def scan_partial_raw_language(
    paths: PipelinePaths,
    language: str,
) -> tuple[tuple[RawShardRef, ...], dict[str, int]]:
    """Scan existing text-only raw shards, truncating corrupt trailing rows.

    This lets resume preserve all fully written documents after crashes.  New
    documents are appended to fresh shard indices; existing shards are never
    appended to, which avoids depending on the last shard's final write state.
    """

    language_dir = raw_language_dir(paths, language)
    refs: list[RawShardRef] = []
    next_indices = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for path in sorted(language_dir.glob(f"{split}-*.jsonl")):
            if path.name.startswith("._"):
                continue
            try:
                index = int(path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            next_indices[split] = max(next_indices[split], index + 1)
            ref = scan_text_jsonl_shard(path, language=language, split=split)
            if ref is None:
                continue
            refs.append(ref)
    refs.sort(key=lambda ref: (ref.language, ref.split, ref.path.name))
    return tuple(refs), next_indices


def scan_text_jsonl_shard(path: Path, *, language: str, split: str) -> RawShardRef | None:
    """Scan and repair one text-only JSONL shard."""

    documents = 0
    estimated_tokens = 0
    valid_offset = 0
    offset = 0
    with path.open("rb+") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line_start = offset
            offset += len(raw)
            if not raw.endswith(b"\n"):
                break
            try:
                row = json.loads(raw)
                text = row.get("text", "") if isinstance(row, Mapping) else ""
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(text, str):
                break
            documents += 1
            estimated_tokens += estimate_document_tokens(text, language=language)
            valid_offset = offset
        if valid_offset < path.stat().st_size:
            f.truncate(valid_offset)
    if documents == 0:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    return RawShardRef(
        path=path,
        language=language,
        split=split,
        documents=documents,
        estimated_tokens=estimated_tokens,
        bytes=valid_offset,
        sha256=file_sha256(path),
    )


def write_raw_language_manifest(
    *,
    config: PipelineConfig,
    paths: PipelinePaths,
    budget: LanguageBudget,
    source_index: int,
    status: str,
    metrics: Mapping[str, object],
    shards: Sequence[RawShardRef],
) -> None:
    """Write per-language raw progress/completion manifest."""

    manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "stage": Stage.COLLECT_RAW.value,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resume_key": raw_language_resume_key(config, budget, source_index),
        "budget": to_jsonable(budget),
        "metrics": to_jsonable(metrics),
        "shards": [to_jsonable(shard) for shard in shards],
    }
    write_json_atomic(raw_language_manifest_path(paths, budget.source.language), manifest)


def raw_shard_from_manifest(row: Mapping[str, object]) -> RawShardRef:
    """Parse a raw shard manifest row."""

    return RawShardRef(
        path=Path(str(row["path"])),
        language=str(row["language"]),
        split=str(row["split"]),
        documents=int(row["documents"]),
        estimated_tokens=int(row["estimated_tokens"]),
        bytes=int(row["bytes"]),
        sha256=str(row["sha256"]),
    )


def collect_raw_languages_multiprocess(
    config: PipelineConfig,
    paths: PipelinePaths,
    pending: Sequence[tuple[int, LanguageBudget]],
    worker_count: int,
) -> list[tuple[int, tuple[RawShardRef, ...], dict[str, object]]]:
    """Collect pending languages in subprocesses and load their completion manifests.

    HF streaming shuffle can leave remote-data retry/cleanup work alive after the
    caller intentionally stops early.  Running each language in its own process
    lets the child bypass Python interpreter cleanup with ``os._exit`` after its
    language manifest is safely on disk, avoiding end-of-run hangs in the parent.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    active: dict[int, tuple[mp.Process, int, LanguageBudget]] = {}
    results: list[tuple[int, tuple[RawShardRef, ...], dict[str, object]]] = []
    next_index = 0

    def start_one(source_index: int, budget: LanguageBudget) -> None:
        error_path = raw_language_error_path(paths, budget.source.language)
        try:
            error_path.unlink()
        except FileNotFoundError:
            pass
        proc = ctx.Process(
            target=collect_raw_language_process_entry,
            args=(config, paths, budget, source_index),
            name=f"fineweb-raw-{budget.source.language}",
        )
        proc.start()
        active[proc.pid or id(proc)] = (proc, source_index, budget)

    while next_index < len(pending) or active:
        while next_index < len(pending) and len(active) < worker_count:
            source_index, budget = pending[next_index]
            start_one(source_index, budget)
            next_index += 1
        finished: list[int] = []
        for key, (proc, source_index, budget) in active.items():
            proc.join(timeout=0)
            if proc.exitcode is None:
                continue
            finished.append(key)
            if proc.exitcode != 0:
                terminate_processes(active.values())
                error_path = raw_language_error_path(paths, budget.source.language)
                error_text = error_path.read_text(encoding="utf-8") if error_path.exists() else ""
                raise RuntimeError(
                    f"raw collection failed for {budget.source.language} "
                    f"with exit code {proc.exitcode}\n{error_text}"
                )
            loaded = load_completed_raw_language(config, paths, budget, source_index)
            if loaded is None:
                terminate_processes(active.values())
                raise RuntimeError(f"raw worker for {budget.source.language} exited without a valid manifest")
            shards, metrics = loaded
            results.append((source_index, shards, metrics))
        for key in finished:
            active.pop(key, None)
        if not finished and active:
            time.sleep(0.2)
    return results


def collect_raw_language_process_entry(
    config: PipelineConfig,
    paths: PipelinePaths,
    budget: LanguageBudget,
    source_index: int,
) -> None:
    """Subprocess entrypoint for one raw language collection."""

    code = 0
    try:
        collect_raw_language(config, paths, budget, source_index)
    except BaseException as exc:  # pragma: no cover - exercised in child process
        code = 1
        error = {
            "language": budget.source.language,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            write_json_atomic(raw_language_error_path(paths, budget.source.language), error)
        except BaseException:
            traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


def raw_language_error_path(paths: PipelinePaths, language: str) -> Path:
    """Return the per-language raw error path."""

    return raw_language_dir(paths, language) / "error.json"


def terminate_processes(processes: Sequence[tuple[mp.Process, int, LanguageBudget]]) -> None:
    """Best-effort termination for active raw worker processes."""

    for proc, _source_index, _budget in processes:
        if proc.exitcode is None:
            proc.terminate()
    for proc, _source_index, _budget in processes:
        proc.join(timeout=5)
        if proc.exitcode is None:
            proc.kill()
            proc.join(timeout=5)


def collect_raw_language(
    config: PipelineConfig,
    paths: PipelinePaths,
    budget: LanguageBudget,
    source_index: int,
) -> tuple[tuple[RawShardRef, ...], dict[str, object]]:
    """Collect one source/language; safe to run concurrently per language."""

    source = budget.source
    existing_shards, next_indices = scan_partial_raw_language(paths, source.language)
    existing_documents = sum(shard.documents for shard in existing_shards)
    if existing_documents > budget.raw_documents:
        raise RuntimeError(
            f"partial raw data for {source.language} has {existing_documents} documents, "
            f"exceeding target {budget.raw_documents}; use --force to restart"
        )
    remaining_documents = budget.raw_documents - existing_documents
    scanned = 0
    newly_kept = 0
    skipped_empty = 0
    skipped_filter = 0
    skipped_existing_replay = 0
    resume_status = "resumed-partial" if existing_documents > 0 and remaining_documents > 0 else "collected"
    if remaining_documents == 0 and existing_documents > 0:
        resume_status = "completed-from-partial"
    print(
        f"collect-raw[{source.language}]: {source.dataset}/{source.config} "
        f"target_docs={budget.raw_documents} existing={existing_documents} "
        f"remaining={remaining_documents}",
        flush=True,
    )

    in_progress_metrics = {
        "language": source.language,
        "dataset": source.dataset,
        "config": source.config,
        "target_documents": budget.raw_documents,
        "existing_documents": existing_documents,
        "remaining_documents": remaining_documents,
        "resume_status": "in-progress",
    }
    write_raw_language_manifest(
        config=config,
        paths=paths,
        budget=budget,
        source_index=source_index,
        status="in-progress",
        metrics=in_progress_metrics,
        shards=existing_shards,
    )

    def documents() -> Iterator[RawDocument]:
        nonlocal scanned, newly_kept, skipped_empty, skipped_filter, skipped_existing_replay
        accepted = 0
        yielded = 0
        records = iter_streaming_source_records(
            source,
            seed=config.seed + source_index,
            shuffle_buffer=config.shuffle_buffer,
        )
        try:
            while yielded < remaining_documents:
                record = next(records)
                scanned += 1
                document = normalize_source_record(source, record)
                if not document.text:
                    skipped_empty += 1
                    continue
                document = replace(
                    document,
                    split=assign_document_split(
                        document,
                        val_fraction=config.val_fraction,
                        seed=config.seed,
                    ),
                )
                if not should_keep_document(document, config):
                    skipped_filter += 1
                    continue
                accepted += 1
                if accepted <= existing_documents:
                    skipped_existing_replay += 1
                    continue
                newly_kept += 1
                yielded += 1
                yield document
        finally:
            close_iterable(records)

    new_shards = write_raw_shards(
        documents(),
        paths,
        budget,
        raw_shard_bytes=config.raw_shard_bytes,
        write_buffer_bytes=config.raw_write_buffer_bytes,
        target_documents=remaining_documents,
        initial_indices=next_indices,
    )
    shards = tuple([*existing_shards, *new_shards])
    kept = existing_documents + newly_kept
    metrics = {
        "language": source.language,
        "dataset": source.dataset,
        "config": source.config,
        "target_documents": budget.raw_documents,
        "scanned_records": scanned,
        "kept_documents": kept,
        "existing_documents": existing_documents,
        "newly_collected_documents": newly_kept,
        "skipped_existing_replay": skipped_existing_replay,
        "skipped_empty": skipped_empty,
        "skipped_filter": skipped_filter,
        "shards": len(shards),
        "train_documents": sum(s.documents for s in shards if s.split == "train"),
        "val_documents": sum(s.documents for s in shards if s.split == "val"),
        "estimated_tokens": sum(s.estimated_tokens for s in shards),
        "bytes": sum(s.bytes for s in shards),
        "resume_status": resume_status,
    }
    write_raw_language_manifest(
        config=config,
        paths=paths,
        budget=budget,
        source_index=source_index,
        status="ok",
        metrics=metrics,
        shards=shards,
    )
    print(
        f"collect-raw[{source.language}]: done kept={kept} existing={existing_documents} "
        f"new={newly_kept} scanned={scanned} shards={len(shards)}",
        flush=True,
    )
    return shards, metrics


def iter_streaming_source_records(source: SourceSpec, *, seed: int, shuffle_buffer: int) -> Iterator[Mapping[str, object]]:
    """Yield raw streaming records from a Hugging Face dataset subset."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without uv deps
        raise RuntimeError(
            "datasets is required for raw collection; run with `uv run tools/data_fineweb2.py ...`"
        ) from exc

    if source.config is None:
        dataset = load_dataset(source.dataset, split=source.split, streaming=True)
    else:
        dataset = load_dataset(source.dataset, source.config, split=source.split, streaming=True)
    if shuffle_buffer > 0:
        # This explicitly asks HF datasets to shuffle the streaming iterable.
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    iterator = iter(dataset)
    try:
        while True:
            yield next(iterator)
    finally:
        close_iterable(iterator)
        close_iterable(dataset)


def normalize_source_record(source: SourceSpec, record: Mapping[str, object]) -> RawDocument:
    """Map one source record to the normalized raw JSONL document schema."""

    raw_text = record.get(source.text_field, "")
    text = raw_text if isinstance(raw_text, str) else ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()

    record_id = ""
    for field_name in source.id_fields:
        value = record.get(field_name)
        if value is not None and str(value):
            record_id = str(value)
            break
    if not record_id:
        record_id = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]

    return RawDocument(
        text=text,
        language=source.language,
        split="",
        source_dataset=source.dataset,
        source_config=source.config,
        source_split=source.split,
        record_id=record_id,
        url=None,
        metadata={},
    )


def should_keep_document(document: RawDocument, config: PipelineConfig) -> bool:
    """Apply document-quality and size filters before raw persistence."""

    n_chars = len(document.text)
    if n_chars < config.min_chars:
        return False
    if config.max_chars > 0 and n_chars > config.max_chars:
        return False
    return True


def assign_document_split(document: RawDocument, *, val_fraction: float, seed: int) -> str:
    """Assign train/val by stable hash so collection is streaming and reproducible."""

    value = stable_hash_int(
        [
            document.language,
            document.source_dataset,
            document.source_config or "",
            document.source_split,
            document.record_id,
        ],
        seed=seed,
    )
    threshold = int(val_fraction * (1 << 64))
    return "val" if value < threshold else "train"


def estimate_document_tokens(text: str, *, language: str) -> int:
    """Estimate tokenizer tokens before the real tokenizer exists."""

    if not text:
        return 0
    if language.startswith(("cmn_", "jpn_")):
        return max(1, int(len(text) * 0.75))
    words = len(text.split())
    if words > 0:
        return max(1, int(words * 1.35))
    return max(1, int(len(text) / 4))


def write_raw_shards(
    documents: Iterator[RawDocument],
    paths: PipelinePaths,
    budget: LanguageBudget,
    *,
    raw_shard_bytes: int = MAX_RAW_SHARD_BYTES,
    write_buffer_bytes: int = DEFAULT_RAW_WRITE_BUFFER_BYTES,
    target_documents: int | None = None,
    initial_indices: Mapping[str, int] | None = None,
) -> tuple[RawShardRef, ...]:
    """Write language/split raw JSONL shards up to the requested collection budget.

    Rows are intentionally text-only: ``{"text": ...}``.  Language/split/source
    metadata lives in the raw manifest and shard paths, which keeps the hot path
    smaller and improves HDD write throughput.
    """

    target = budget.raw_documents if target_documents is None else target_documents
    if target < 0:
        raise ValueError("target_documents must be non-negative")
    language_dir = paths.raw_dir / budget.source.language
    language_dir.mkdir(parents=True, exist_ok=True)

    class WriterState:
        def __init__(self, split: str):
            self.split = split
            self.index = int((initial_indices or {}).get(split, 0))
            self.path: Path | None = None
            self.handle: BinaryIO | None = None
            self.buffer = bytearray()
            self.documents = 0
            self.estimated_tokens = 0
            self.bytes = 0

    states: dict[str, WriterState] = {}
    completed: list[dict[str, object]] = []

    def flush_state(state: WriterState) -> None:
        if state.handle is not None and state.buffer:
            state.handle.write(state.buffer)
            state.buffer.clear()

    def open_state(state: WriterState) -> None:
        state.path = language_dir / f"{state.split}-{state.index:05d}.jsonl"
        state.handle = state.path.open("wb")
        state.buffer.clear()
        state.documents = 0
        state.estimated_tokens = 0
        state.bytes = 0

    def close_state(state: WriterState) -> None:
        if state.handle is None or state.path is None:
            return
        flush_state(state)
        state.handle.close()
        if state.documents > 0:
            completed.append(
                {
                    "path": state.path,
                    "language": budget.source.language,
                    "split": state.split,
                    "documents": state.documents,
                    "estimated_tokens": state.estimated_tokens,
                    "bytes": state.bytes,
                }
            )
        elif state.path.exists():
            state.path.unlink()
        state.index += 1
        state.path = None
        state.handle = None
        state.buffer.clear()
        state.documents = 0
        state.estimated_tokens = 0
        state.bytes = 0

    def get_state(split: str) -> WriterState:
        if split not in states:
            states[split] = WriterState(split)
            open_state(states[split])
        return states[split]

    written = 0
    try:
        for document in documents:
            if written >= target:
                break
            split = document.split or "train"
            if split not in {"train", "val"}:
                raise ValueError(f"invalid document split {split!r}")
            encoded = (json.dumps(raw_document_to_row(document), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            state = get_state(split)
            if state.documents > 0 and state.bytes + len(encoded) > raw_shard_bytes:
                close_state(state)
                open_state(state)
            state.buffer.extend(encoded)
            state.documents += 1
            state.estimated_tokens += estimate_document_tokens(document.text, language=document.language)
            state.bytes += len(encoded)
            written += 1
            if len(state.buffer) >= write_buffer_bytes:
                flush_state(state)
    finally:
        for state in states.values():
            close_state(state)

    if written < target:
        raise RuntimeError(
            f"source {budget.source.language} exhausted before target: "
            f"wrote {written}, target {target}"
        )

    refs = []
    for item in completed:
        path = item["path"]
        assert isinstance(path, Path)
        refs.append(
            RawShardRef(
                path=path,
                language=str(item["language"]),
                split=str(item["split"]),
                documents=int(item["documents"]),
                estimated_tokens=int(item["estimated_tokens"]),
                bytes=int(item["bytes"]),
                sha256=file_sha256(path),
            )
        )
    return tuple(refs)


def discover_raw_shards(paths: PipelinePaths, *, language: str | None = None, split: str | None = None) -> tuple[RawShardRef, ...]:
    """Read raw shard metadata for downstream stages.

    Prefer the aggregate raw manifest, but fall back to per-language manifests so
    a run that completed languages but died before the aggregate write can still
    be resumed without redownloading.
    """

    manifest_path = paths.raw_dir / "manifest.json"
    rows: list[Mapping[str, object]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(manifest.get("shards", []))
    else:
        for language_manifest in sorted(paths.raw_dir.glob("*/manifest.json")):
            manifest = json.loads(language_manifest.read_text(encoding="utf-8"))
            if manifest.get("stage") == Stage.COLLECT_RAW.value and manifest.get("status") == "ok":
                rows.extend(manifest.get("shards", []))
    if not rows:
        raise FileNotFoundError(manifest_path)
    shards = []
    for row in rows:
        shard = raw_shard_from_manifest(row)
        if language is not None and shard.language != language:
            continue
        if split is not None and shard.split != split:
            continue
        shards.append(shard)
    return tuple(shards)


# ---------------------------------------------------------------------------
# Stage: tokenizer training corpus and tokenizer training
# ---------------------------------------------------------------------------


def build_tokenizer_corpus_stage(config: PipelineConfig, paths: PipelinePaths, budgets: Sequence[LanguageBudget]) -> StageSummary:
    """Create a stratified tokenizer-training JSONL sample from train-only raw docs."""

    if paths.tokenizer_train_path.exists() and config.resume and not config.force:
        metrics_path = paths.tokenizer_train_path.with_suffix(".manifest.json")
        metrics: Mapping[str, object] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("metrics", {})
        return StageSummary(
            stage=Stage.BUILD_TOKENIZER_CORPUS,
            status="skipped-existing",
            metrics=metrics,
            artifacts=(paths.tokenizer_train_path,),
        )
    if paths.tokenizer_train_path.exists() and not config.force:
        raise FileExistsError(paths.tokenizer_train_path)

    raw_shards = discover_raw_shards(paths, split="train")
    if not raw_shards:
        raise ValueError("no train raw shards found; run collect-raw first")
    documents = iter_tokenizer_training_documents(raw_shards, budgets)
    summary = write_tokenizer_training_jsonl(documents, paths.tokenizer_train_path)

    metrics_manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "stage": Stage.BUILD_TOKENIZER_CORPUS.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": to_jsonable(config),
        "budgets": to_jsonable(budgets),
        "metrics": summary.metrics,
        "artifacts": [str(path) for path in summary.artifacts],
    }
    metrics_path = paths.tokenizer_train_path.with_suffix(".manifest.json")
    metrics_path.write_text(json.dumps(metrics_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return StageSummary(
        stage=Stage.BUILD_TOKENIZER_CORPUS,
        status=summary.status,
        metrics=summary.metrics,
        artifacts=tuple([*summary.artifacts, metrics_path]),
    )


def iter_tokenizer_training_documents(raw_shards: Sequence[RawShardRef], budgets: Sequence[LanguageBudget]) -> Iterator[RawDocument]:
    """Yield train-split documents selected for tokenizer training.

    Selection is stratified by language and bounded by each language budget's
    estimated-token target.  Only train raw shards are eligible, so validation
    documents are not used for tokenizer training.
    """

    target_by_language = {
        budget.source.language: budget.tokenizer_train_estimated_tokens for budget in budgets
    }
    used_by_language = {language: 0 for language in target_by_language}
    exhausted_by_language: set[str] = set()
    for shard in sorted(raw_shards, key=lambda s: (s.language, s.split, str(s.path))):
        if shard.split != "train":
            continue
        target = target_by_language.get(shard.language)
        if target is None or shard.language in exhausted_by_language:
            continue
        for row in read_jsonl_rows(shard.path):
            text_value = row.get("text", "")
            text = text_value if isinstance(text_value, str) else ""
            if not text:
                continue
            estimated = estimate_document_tokens(text, language=shard.language)
            if used_by_language[shard.language] >= target:
                exhausted_by_language.add(shard.language)
                break
            used_by_language[shard.language] += estimated
            yield RawDocument(
                text=text,
                language=shard.language,
                split="train",
                source_dataset="",
                source_config=None,
                source_split="train",
                record_id="",
            )


def write_tokenizer_training_jsonl(documents: Iterator[RawDocument], path: Path) -> StageSummary:
    """Persist the tokenizer-training sample as JSONL with a ``text`` field."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    docs = 0
    estimated_tokens = 0
    bytes_written = 0
    by_language: dict[str, dict[str, int]] = {}
    buffer = bytearray()
    try:
        with tmp_path.open("wb") as f:
            for document in documents:
                encoded = (json.dumps({"text": document.text}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                buffer.extend(encoded)
                docs += 1
                est = estimate_document_tokens(document.text, language=document.language)
                estimated_tokens += est
                bytes_written += len(encoded)
                lang_metrics = by_language.setdefault(document.language, {"documents": 0, "estimated_tokens": 0, "bytes": 0})
                lang_metrics["documents"] += 1
                lang_metrics["estimated_tokens"] += est
                lang_metrics["bytes"] += len(encoded)
                if len(buffer) >= DEFAULT_RAW_WRITE_BUFFER_BYTES:
                    f.write(buffer)
                    buffer.clear()
            if buffer:
                f.write(buffer)
                buffer.clear()
        tmp_path.replace(path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    if docs == 0:
        raise ValueError("tokenizer training corpus is empty")
    metrics = {
        "documents": docs,
        "estimated_tokens": estimated_tokens,
        "bytes": bytes_written,
        "sha256": file_sha256(path),
        "languages": by_language,
    }
    return StageSummary(
        stage=Stage.BUILD_TOKENIZER_CORPUS,
        status="ok",
        metrics=metrics,
        artifacts=(path,),
    )


def train_tokenizer_stage(config: PipelineConfig, paths: PipelinePaths) -> StageSummary:
    """Train the shared tokenizer with the streamed JSONL tokenizer helper."""

    if not paths.tokenizer_train_path.exists():
        raise FileNotFoundError(f"tokenizer training corpus not found: {paths.tokenizer_train_path}")
    if paths.tokenizer_path.exists() and config.resume and not config.force:
        tokenizer = load_tokenizer_ref(paths.tokenizer_path)
        return StageSummary(
            stage=Stage.TRAIN_TOKENIZER,
            status="skipped-existing",
            metrics=to_jsonable(tokenizer),
            artifacts=(paths.tokenizer_path,),
        )
    if paths.tokenizer_path.exists() and not config.force:
        raise FileExistsError(paths.tokenizer_path)
    if config.force and paths.tokenizer_path.exists():
        paths.tokenizer_path.unlink()

    from tok_utils import default_tokenizer_binary, train_tokenizer_jsonl

    binary = default_tokenizer_binary()
    if not binary.exists():
        raise FileNotFoundError(f"tokenizer binary not found: {binary}; run `make tools`")
    summary = train_tokenizer_jsonl(
        tokenizer=paths.tokenizer_path,
        input_jsonl=paths.tokenizer_train_path,
        jsonl_text_field="text",
        im_start=IM_START,
        im_end=IM_END,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=(PAD,),
        seed=config.seed,
        binary=binary,
    )
    tokenizer = load_tokenizer_ref(paths.tokenizer_path)
    metrics = {
        "summary": dict(summary),
        "tokenizer": to_jsonable(tokenizer),
        "tokenizer_sha256": file_sha256(paths.tokenizer_path),
    }
    return StageSummary(
        stage=Stage.TRAIN_TOKENIZER,
        status="ok",
        metrics=metrics,
        artifacts=(paths.tokenizer_path,),
    )


def load_tokenizer_ref(tokenizer_path: Path) -> TokenizerRef:
    """Load tokenizer ids/hash needed by tokenization, packing, and manifest writing."""

    with tokenizer_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("format") != "gd-bpe-tokenizer-v1":
        raise ValueError(f"unsupported tokenizer format in {tokenizer_path}")
    token_ids: dict[str, int] = {}
    for token in spec.get("tokens", []):
        if isinstance(token, Mapping) and token.get("kind") == "special":
            token_ids[str(token.get("text"))] = int(token["id"])
    missing = [token for token in (PAD, IM_START, IM_END) if token not in token_ids]
    if missing:
        raise ValueError(f"tokenizer {tokenizer_path} missing special tokens: {missing}")
    raw_hash = spec.get("hash", "0")
    if isinstance(raw_hash, str):
        tokenizer_hash = int(raw_hash, 16)
    else:
        tokenizer_hash = int(raw_hash)
    return TokenizerRef(
        path=tokenizer_path,
        vocab_size=int(spec.get("vocab_size", 0)),
        hash=tokenizer_hash,
        pad_id=token_ids[PAD],
        im_start_id=token_ids[IM_START],
        im_end_id=token_ids[IM_END],
    )


# ---------------------------------------------------------------------------
# Stage: tokenize raw JSONL shards
# ---------------------------------------------------------------------------


def tokenize_stage(config: PipelineConfig, paths: PipelinePaths, tokenizer: TokenizerRef) -> StageSummary:
    """Tokenize every raw JSONL shard with the shared tokenizer."""

    manifest_path = paths.tokenized_dir / "manifest.json"
    if manifest_path.exists() and config.resume and not config.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return StageSummary(
            stage=Stage.TOKENIZE,
            status="skipped-existing",
            metrics=manifest.get("metrics", {}),
            artifacts=(manifest_path,),
        )
    if manifest_path.exists() and not config.force:
        raise FileExistsError(manifest_path)

    raw_shards = discover_raw_shards(paths)
    if not raw_shards:
        raise ValueError("no raw shards found; run collect-raw first")
    max_workers = min(config.workers, len(raw_shards))
    print(
        f"tokenize: raw_shards={len(raw_shards)} workers={max_workers} "
        f"tokenizer={tokenizer.path}",
        flush=True,
    )

    results: list[tuple[int, TokenizedShardRef]] = []
    if max_workers == 1:
        for index, raw_shard in enumerate(raw_shards):
            results.append((index, tokenize_raw_shard(raw_shard, tokenizer, paths, config)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fineweb-tokenize") as pool:
            futures = {
                pool.submit(tokenize_raw_shard, raw_shard, tokenizer, paths, config): index
                for index, raw_shard in enumerate(raw_shards)
            }
            for future in as_completed(futures):
                index = futures[future]
                results.append((index, future.result()))

    results.sort(key=lambda item: item[0])
    refs = [ref for _index, ref in results]
    metrics = {
        "raw_shards": len(raw_shards),
        "tokenized_shards": len(refs),
        "documents": sum(ref.documents for ref in refs),
        "tokens": sum(ref.tokens for ref in refs),
        "languages": sorted({ref.language for ref in refs}),
        "workers": max_workers,
        "tokenizer": to_jsonable(tokenizer),
    }
    manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "stage": Stage.TOKENIZE.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": to_jsonable(config),
        "tokenizer": to_jsonable(tokenizer),
        "metrics": metrics,
        "shards": [to_jsonable(ref) for ref in refs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return StageSummary(
        stage=Stage.TOKENIZE,
        status="ok",
        metrics=metrics,
        artifacts=tuple([manifest_path, *[ref.path for ref in refs]]),
    )


def tokenize_raw_shard(raw_shard: RawShardRef, tokenizer: TokenizerRef, paths: PipelinePaths, config: PipelineConfig) -> TokenizedShardRef:
    """Run gradients-tokenize on one raw shard and return tokenized metadata."""

    from tok_utils import TokenizedCorpus, default_tokenizer_binary, tokenize_jsonl

    binary = default_tokenizer_binary()
    if not binary.exists():
        raise FileNotFoundError(f"tokenizer binary not found: {binary}; run `make tools`")
    output_dir = paths.tokenized_dir / raw_shard.language / raw_shard.split / raw_shard.path.stem
    if output_dir.exists():
        if config.force:
            shutil.rmtree(output_dir, ignore_errors=True)
        elif config.resume:
            corpus = TokenizedCorpus.open(output_dir)
            return TokenizedShardRef(
                path=output_dir,
                source_raw_shard=raw_shard.path,
                language=raw_shard.language,
                split=raw_shard.split,
                documents=corpus.num_sequences,
                tokens=corpus.num_tokens,
            )
        else:
            raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"tokenize[{raw_shard.language}/{raw_shard.split}]: {raw_shard.path.name}",
        flush=True,
    )
    summary = tokenize_jsonl(
        tokenizer=tokenizer.path,
        input_jsonl=raw_shard.path,
        output_dir=output_dir,
        im_start=IM_START,
        im_end=IM_END,
        max_length=2_147_483_647,
        jsonl_text_field="text",
        use_tokenizer=True,
        binary=binary,
    )
    corpus = TokenizedCorpus.open(output_dir)
    if corpus.mode != "jsonl":
        raise ValueError(f"expected JSONL tokenized corpus at {output_dir}, got {corpus.mode!r}")
    if corpus.im_start_id != tokenizer.im_start_id or corpus.im_end_id != tokenizer.im_end_id:
        raise ValueError(f"special-token id mismatch in tokenized shard {output_dir}")
    sequences = int(summary.get("sequences", corpus.num_sequences))
    tokens = int(summary.get("tokens", corpus.num_tokens))
    if sequences != corpus.num_sequences or tokens != corpus.num_tokens:
        raise ValueError(f"tokenized summary/manifest mismatch for {output_dir}")
    return TokenizedShardRef(
        path=output_dir,
        source_raw_shard=raw_shard.path,
        language=raw_shard.language,
        split=raw_shard.split,
        documents=corpus.num_sequences,
        tokens=corpus.num_tokens,
    )


def discover_tokenized_shards(paths: PipelinePaths, *, language: str | None = None, split: str | None = None) -> tuple[TokenizedShardRef, ...]:
    """Read or reconstruct tokenized shard metadata for packing."""

    manifest_path = paths.tokenized_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = []
    for row in manifest.get("shards", []):
        ref = TokenizedShardRef(
            path=Path(row["path"]),
            source_raw_shard=Path(row["source_raw_shard"]),
            language=str(row["language"]),
            split=str(row["split"]),
            documents=int(row["documents"]),
            tokens=int(row["tokens"]),
        )
        if language is not None and ref.language != language:
            continue
        if split is not None and ref.split != split:
            continue
        refs.append(ref)
    return tuple(refs)


# ---------------------------------------------------------------------------
# Stage: streaming pack to GDDS
# ---------------------------------------------------------------------------


def pack_stage(config: PipelineConfig, paths: PipelinePaths, budgets: Sequence[LanguageBudget], tokenizer: TokenizerRef) -> StageSummary:
    """Pack tokenized documents into fixed-length GDDS LM samples."""

    manifest_path = paths.gdds_dir / "manifest.json"
    if manifest_path.exists() and config.resume and not config.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return StageSummary(
            stage=Stage.PACK,
            status="skipped-existing",
            metrics=manifest.get("metrics", {}),
            artifacts=(manifest_path,),
        )
    if manifest_path.exists() and not config.force:
        raise FileExistsError(manifest_path)
    if config.force:
        remove_gdds_split(paths.gdds_dir, "train")
        remove_gdds_split(paths.gdds_dir, "val")
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass

    tokenized_refs = discover_tokenized_shards(paths)
    train_refs = tuple(ref for ref in tokenized_refs if ref.split == "train")
    val_refs = tuple(ref for ref in tokenized_refs if ref.split == "val")
    if not train_refs:
        raise ValueError("no tokenized train shards found; run tokenize first")

    target_train_samples = sum(budget.train_packed_samples for budget in budgets)
    target_val_samples = sum(budget.val_packed_samples for budget in budgets)
    print(
        f"pack: train_shards={len(train_refs)} val_shards={len(val_refs)} "
        f"context={config.context_length} max_shard_bytes={config.max_shard_bytes}",
        flush=True,
    )

    train_packed = write_gdds_split(
        iter_packed_samples(
            train_refs,
            context_length=config.context_length,
            split="train",
            max_samples=target_train_samples,
        ),
        paths,
        split="train",
        max_shard_bytes=config.max_shard_bytes,
    )
    val_packed: tuple[PackedShardRef, ...] = ()
    if val_refs:
        val_packed = write_gdds_split(
            iter_packed_samples(
                val_refs,
                context_length=config.context_length,
                split="val",
                max_samples=target_val_samples,
            ),
            paths,
            split="val",
            max_shard_bytes=config.max_shard_bytes,
        )

    packed_refs = tuple([*train_packed, *val_packed])
    if not train_packed:
        raise ValueError("packing produced no train GDDS shards")
    storage_fields = storage_fields_for_context(config.context_length)
    runtime_fields = runtime_fields_for_context(config.context_length)
    from gdds_utils import field_metadata, schema_hash

    train_samples = sum(ref.samples for ref in train_packed)
    val_samples = sum(ref.samples for ref in val_packed)
    metrics = {
        "context_length": config.context_length,
        "record_length": config.context_length + 1,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_packed_tokens": train_samples * config.context_length,
        "val_packed_tokens": val_samples * config.context_length,
        "target_train_samples": target_train_samples,
        "target_val_samples": target_val_samples,
        "target_train_reached": train_samples >= target_train_samples,
        "target_val_reached": val_samples >= target_val_samples if val_refs else target_val_samples == 0,
        "tokenized_train_tokens": sum(ref.tokens for ref in train_refs),
        "tokenized_val_tokens": sum(ref.tokens for ref in val_refs),
        "shards": len(packed_refs),
        "storage_schema_hash": f"0x{schema_hash(storage_fields):016x}",
        "runtime_schema_hash": f"0x{schema_hash(runtime_fields):016x}",
        "tokenizer": to_jsonable(tokenizer),
    }
    manifest = {
        "format_version": DATA_FORMAT_VERSION,
        "stage": Stage.PACK.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": to_jsonable(config),
        "tokenizer": to_jsonable(tokenizer),
        "metrics": metrics,
        "storage_fields": [field_metadata(field) for field in storage_fields],
        "runtime_fields": [field_metadata(field) for field in runtime_fields],
        "shards": [to_jsonable(ref) for ref in packed_refs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return StageSummary(
        stage=Stage.PACK,
        status="ok",
        metrics=metrics,
        artifacts=tuple([manifest_path, *[ref.path for ref in packed_refs]]),
    )


def storage_fields_for_context(context_length: int) -> object:
    """Return compact on-disk GDDS fields: tokens and segment_lengths."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    from gdds_utils import FieldSpec

    return [
        FieldSpec("tokens", "u16", (context_length + 1,), collate="stack"),
        FieldSpec("segment_lengths", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
    ]


def runtime_fields_for_context(context_length: int) -> object:
    """Return runtime GDDS fields consumed by examples/gpt_lm."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    from gdds_utils import FieldSpec

    return [
        FieldSpec("input_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec("positions", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec("target_ids", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec("segment_lengths", "i32", (-1,), collate="packed_sequence", ragged_dim=0),
        FieldSpec(
            "cu_seqlens",
            "i32",
            (-1,),
            collate="generated",
            generated="cu_seqlens_from_lengths",
            source="segment_lengths",
        ),
    ]


def iter_tokenized_documents(tokenized_shards: Sequence[TokenizedShardRef]) -> Iterator[tuple[object, str]]:
    """Yield ``(token_ids, language)`` document sequences from tokenized shard dirs."""

    import numpy as np
    from tok_utils import TOKEN_DTYPE, TokenizedCorpus

    for ref in sorted(tokenized_shards, key=lambda r: (r.language, r.split, str(r.path))):
        corpus = TokenizedCorpus.open(ref.path)
        if corpus.mode != "jsonl":
            raise ValueError(f"expected jsonl tokenized corpus at {ref.path}, got {corpus.mode!r}")
        for seq in corpus.iter_sequences(copy=False):
            arr = np.asarray(seq, dtype=TOKEN_DTYPE)
            if arr.size > 0:
                yield arr, ref.language


def iter_packed_samples(
    tokenized_shards: Sequence[TokenizedShardRef],
    *,
    context_length: int,
    split: str,
    max_samples: int | None = None,
) -> Iterator[PackedSample]:
    """Streaming fixed-window packer that never concatenates the full corpus."""

    import numpy as np
    from tok_utils import TOKEN_DTYPE

    if context_length < 2:
        raise ValueError("context_length must be >= 2")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    record_length = context_length + 1
    pending_tokens = np.empty((0,), dtype=TOKEN_DTYPE)
    pending_doc_ids = np.empty((0,), dtype=np.int64)
    doc_index = 0
    emitted = 0
    for doc_tokens, _language in iter_tokenized_documents(tokenized_shards):
        tokens = np.asarray(doc_tokens, dtype=TOKEN_DTYPE)
        doc_ids = np.full(tokens.shape, doc_index, dtype=np.int64)
        doc_index += 1
        if pending_tokens.size == 0:
            pending_tokens = tokens
            pending_doc_ids = doc_ids
        else:
            pending_tokens = np.concatenate([pending_tokens, tokens])
            pending_doc_ids = np.concatenate([pending_doc_ids, doc_ids])
        while pending_tokens.size >= record_length:
            if max_samples is not None and emitted >= max_samples:
                return
            row = np.asarray(pending_tokens[:record_length], dtype=TOKEN_DTYPE)
            row_doc_ids = np.asarray(pending_doc_ids[:record_length], dtype=np.int64)
            segment_lengths = compute_segment_lengths(row_doc_ids[:-1])
            if int(segment_lengths.sum()) != context_length:
                raise ValueError("segment lengths do not cover packed context")
            yield PackedSample(tokens=row, segment_lengths=segment_lengths, language="mixed", split=split)
            emitted += 1
            pending_tokens = pending_tokens[context_length:]
            pending_doc_ids = pending_doc_ids[context_length:]


def compute_segment_lengths(doc_ids: object) -> object:
    """Compute document-fragment run lengths over ``tokens[:-1]`` for one sample."""

    import numpy as np
    from tok_utils import TOKEN_DTYPE

    arr = np.asarray(doc_ids)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("doc_ids must be a non-empty vector")
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    bounds = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            change.astype(np.int64, copy=False),
            np.asarray([arr.size], dtype=np.int64),
        ]
    )
    return np.diff(bounds).astype(TOKEN_DTYPE, copy=False)


def write_gdds_split(samples: Iterator[PackedSample], paths: PipelinePaths, *, split: str, max_shard_bytes: int) -> tuple[PackedShardRef, ...]:
    """Serialize packed samples for one split to GDDS shards."""

    from gdds_utils import GddsSplitWriter, schema_hash

    first_sample: PackedSample | None = None
    iterator = iter(samples)
    try:
        first_sample = next(iterator)
    except StopIteration:
        return ()
    context_length = int(len(first_sample.tokens) - 1)
    fields = storage_fields_for_context(context_length)
    writer = GddsSplitWriter(paths.gdds_dir, split, fields, max_shard_bytes=max_shard_bytes)
    try:
        writer.write_sample(packed_sample_to_gdds(first_sample))
        for sample in iterator:
            writer.write_sample(packed_sample_to_gdds(sample))
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    schema_hash_value = schema_hash(fields)
    refs = []
    for info in writer.shard_infos:
        refs.append(
            PackedShardRef(
                path=info.path,
                language="mixed",
                split=split,
                samples=info.samples,
                packed_tokens=info.samples * context_length,
                schema_hash=schema_hash_value,
                bytes=info.file_nbytes,
                sha256=file_sha256(info.path),
            )
        )
    return tuple(refs)


def discover_packed_shards(paths: PipelinePaths, *, language: str | None = None, split: str | None = None) -> tuple[PackedShardRef, ...]:
    """Read or reconstruct final GDDS shard metadata."""

    manifest_path = paths.gdds_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = []
    for row in manifest.get("shards", []):
        ref = PackedShardRef(
            path=Path(row["path"]),
            language=str(row["language"]),
            split=str(row["split"]),
            samples=int(row["samples"]),
            packed_tokens=int(row["packed_tokens"]),
            schema_hash=int(row["schema_hash"]),
            bytes=int(row["bytes"]),
            sha256=str(row["sha256"]),
        )
        if language is not None and ref.language != language:
            continue
        if split is not None and ref.split != split:
            continue
        refs.append(ref)
    return tuple(refs)


# ---------------------------------------------------------------------------
# Stage: verification and manifest
# ---------------------------------------------------------------------------


def verify_stage(config: PipelineConfig, paths: PipelinePaths, tokenizer: TokenizerRef) -> StageSummary:
    """Verify tokenizer ids, packed counts, GDDS schemas, and sample invariants."""

    raise NotImplementedError("verify_stage")


def verify_raw_corpus(raw_shards: Sequence[RawShardRef], budgets: Sequence[LanguageBudget]) -> None:
    """Validate raw shard counts and budget coverage."""

    raise NotImplementedError("verify_raw_corpus")


def verify_tokenized_corpus(tokenized_shards: Sequence[TokenizedShardRef], tokenizer: TokenizerRef) -> None:
    """Validate tokenized shard manifests and special-token ids."""

    raise NotImplementedError("verify_tokenized_corpus")


def verify_gdds_shards(packed_shards: Sequence[PackedShardRef], config: PipelineConfig, tokenizer: TokenizerRef) -> None:
    """Validate final GDDS samples, schema hashes, token id ranges, and counts."""

    raise NotImplementedError("verify_gdds_shards")


def preview_decoded_samples(packed_shards: Sequence[PackedShardRef], tokenizer: TokenizerRef, *, count: int, seed: int) -> None:
    """Decode random packed rows for human inspection."""

    raise NotImplementedError("preview_decoded_samples")


def write_manifest_stage(config: PipelineConfig, paths: PipelinePaths, summaries: Sequence[StageSummary]) -> StageSummary:
    """Write the final dataset manifest and split/source indexes."""

    raise NotImplementedError("write_manifest_stage")


def build_manifest(config: PipelineConfig, paths: PipelinePaths, budgets: Sequence[LanguageBudget], tokenizer: TokenizerRef, summaries: Sequence[StageSummary]) -> Mapping[str, object]:
    """Assemble the manifest object before JSON serialization."""

    raise NotImplementedError("build_manifest")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def load_project_dotenv() -> None:
    """Load .env from the working directory or one of its parents.

    This makes HF_TOKEN/HF_HOME/HF_DATASETS_CACHE available without requiring
    callers to `source .env` before launching the script. Existing environment
    variables still win over .env values.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for directory in (Path.cwd(), *Path.cwd().parents):
        dotenv_path = directory / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return


def pipeline_cli_summary(summary: PipelineSummary) -> Mapping[str, object]:
    """Return a concise command-line summary without dumping every shard path."""

    return {
        "out_root": summary.paths.out_root,
        "manifest_path": summary.manifest_path,
        "stages": [
            {
                "stage": stage.stage,
                "status": stage.status,
                "metrics": stage.metrics,
                "artifacts": len(stage.artifacts),
            }
            for stage in summary.stages
        ],
    }


def remove_gdds_split(directory: Path, split: str) -> None:
    """Remove existing GDDS shards for a split."""

    if not directory.exists():
        return
    for path in directory.glob(f"{split}-*.gdds"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for path in directory.glob(f"{split}-*.gdds.tmp"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def packed_sample_to_gdds(sample: PackedSample) -> Mapping[str, object]:
    """Convert one packed sample into GDDS TensorData fields."""

    return {
        "tokens": u16_tensor(sample.tokens),
        "segment_lengths": i32_tensor(sample.segment_lengths),
    }


def i32_tensor(array: object) -> object:
    """Return a GDDS i32 TensorData vector."""

    import numpy as np
    from gdds_utils import TensorData

    arr = np.asarray(array, dtype=np.dtype("<i4"))
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return TensorData(dtype="i32", shape=(int(arr.size),), data=arr.tobytes(order="C"))


def u16_tensor(array: object) -> object:
    """Return a GDDS u16 TensorData vector, validating token id range."""

    import numpy as np
    from gdds_utils import TensorData

    arr_i64 = np.asarray(array, dtype=np.int64)
    if arr_i64.ndim != 1:
        arr_i64 = arr_i64.reshape(-1)
    if arr_i64.size == 0:
        raise ValueError("token rows cannot be empty")
    if int(arr_i64.min()) < 0 or int(arr_i64.max()) > 0xFFFF:
        raise ValueError("token id does not fit in uint16 compact storage")
    arr = arr_i64.astype(np.dtype("<u2"), copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return TensorData(dtype="u16", shape=(int(arr.size),), data=arr.tobytes(order="C"))


def close_iterable(value: object) -> None:
    """Best-effort close for HF streaming iterators/generators stopped early."""

    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def stable_hash_int(parts: Sequence[object], *, seed: int) -> int:
    """Return a deterministic integer hash for splitting/sampling decisions."""

    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8", errors="replace"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def file_sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    """Write JSON by replacing a temporary file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def write_jsonl_row(path: Path, row: Mapping[str, object]) -> None:
    """Append one JSON object as UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def read_jsonl_rows(path: Path) -> Iterator[Mapping[str, object]]:
    """Yield JSONL rows from disk."""

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {line_no} in {path} is not an object")
            yield row


def parse_count(text: str | int) -> int:
    """Parse integers with optional k/m/g suffixes."""

    if isinstance(text, int):
        return text
    s = str(text).strip().replace("_", "").lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def parse_csv(text: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI string."""

    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_stage_list(text: str) -> tuple[Stage, ...]:
    """Parse comma-separated stage names."""

    stages = []
    for part in parse_csv(text):
        try:
            stages.append(Stage(part))
        except ValueError as exc:
            known = ", ".join(stage.value for stage in Stage)
            raise argparse.ArgumentTypeError(f"unknown stage {part!r}; expected one of: {known}") from exc
    return tuple(stages)


def raw_document_to_row(document: RawDocument) -> Mapping[str, object]:
    """Convert a normalized document into the persisted raw JSONL schema."""

    return {"text": document.text}


def make_jsonable_scalar(value: object) -> object | None:
    """Keep simple JSON-friendly metadata values and drop everything else."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        out = [make_jsonable_scalar(item) for item in value]
        return [item for item in out if item is not None]
    if isinstance(value, Mapping):
        return {
            str(key): jsonable_value
            for key, item in value.items()
            if (jsonable_value := make_jsonable_scalar(item)) is not None
        }
    return str(value)


def to_jsonable(value: object) -> object:
    """Convert dataclasses, enums, paths, and containers into JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


__all__ = [
    "DATA_FORMAT_VERSION",
    "DEFAULT_SOURCE_SPECS",
    "IM_END",
    "IM_START",
    "PAD",
    "LanguageBudget",
    "PackedSample",
    "PackedShardRef",
    "PipelineConfig",
    "PipelinePaths",
    "PipelineSummary",
    "RawDocument",
    "RawShardRef",
    "SourceSpec",
    "Stage",
    "StageSummary",
    "TokenizerRef",
    "TokenizedShardRef",
    "build_manifest",
    "build_tokenizer_corpus_stage",
    "collect_raw_stage",
    "compute_language_budgets",
    "config_from_args",
    "main",
    "pack_stage",
    "parse_args",
    "run_pipeline",
    "tokenize_stage",
    "train_tokenizer_stage",
    "verify_stage",
    "write_manifest_stage",
]


if __name__ == "__main__":
    raise SystemExit(main())
