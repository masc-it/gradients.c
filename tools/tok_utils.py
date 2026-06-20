"""Utilities for gradients.c tokenized corpora.

The C tokenizer writes mmap-friendly shard directories with a manifest.  This
module intentionally keeps the Python side thin: dataset-specific scripts should
use the C tokenizer for tokenization and these helpers for launching it and
reading the resulting shards via NumPy memmap.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

TOKENIZED_CORPUS_FORMAT = "gd-tokenized-corpus-v1"
TOKEN_DTYPE = np.dtype("<i4")
OFFSET_DTYPE = np.dtype("<u8")


class TokenizerToolError(RuntimeError):
    """Raised when the C tokenizer tool fails or emits invalid output."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_tokenizer_binary() -> Path:
    """Return the default in-tree gradients-tokenize binary path."""

    return _repo_root() / "build" / "tools" / "gradients-tokenize"


def _run(args: Sequence[str | Path]) -> Mapping[str, object]:
    cmd = [str(a) for a in args]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise TokenizerToolError(
            "tokenizer command failed\n"
            f"  command: {' '.join(cmd)}\n"
            f"  returncode: {proc.returncode}\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}"
        )
    stdout = proc.stdout.strip()
    if not stdout:
        return {}
    last_line = stdout.splitlines()[-1]
    try:
        out = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise TokenizerToolError(f"tokenizer emitted non-JSON summary: {last_line!r}") from exc
    if not isinstance(out, dict):
        raise TokenizerToolError(f"tokenizer summary is not an object: {out!r}")
    return out


def _append_training_options(
    args: list[str | Path],
    *,
    vocab_size: int,
    min_frequency: int,
    special_tokens: Sequence[str],
    seed: int,
) -> None:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    args += ["--vocab-size", str(vocab_size), "--min-frequency", str(min_frequency), "--seed", str(seed)]
    for token in special_tokens:
        args += ["--special", token]


def _append_optional_im_tokens(args: list[str | Path], im_start: str | None, im_end: str | None) -> None:
    if (im_start is None) != (im_end is None):
        raise ValueError("im_start and im_end must be provided together")
    if im_start is not None and im_end is not None:
        args += ["--im-start", im_start, "--im-end", im_end]


def train_tokenizer_file(
    *,
    tokenizer: str | Path,
    input_path: str | Path,
    im_start: str | None = None,
    im_end: str | None = None,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = (),
    seed: int = 17,
    binary: str | Path | None = None,
) -> Mapping[str, object]:
    """Train and save a BPE tokenizer from a text file without tokenized output.

    The native trainer streams input into a global pretoken-count table instead
    of loading the whole corpus into memory. Optional ``im_start``/``im_end``
    values are reserved as special tokens for later document-delimited LM use.
    """

    exe = Path(binary) if binary is not None else default_tokenizer_binary()
    args: list[str | Path] = [exe, "--tokenizer", tokenizer, "--input", input_path, "--train-only"]
    _append_optional_im_tokens(args, im_start, im_end)
    _append_training_options(
        args,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        seed=seed,
    )
    return _run(args)


def train_tokenizer_jsonl(
    *,
    tokenizer: str | Path,
    input_jsonl: str | Path,
    jsonl_text_field: str = "text",
    im_start: str | None = None,
    im_end: str | None = None,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = (),
    seed: int = 17,
    binary: str | Path | None = None,
) -> Mapping[str, object]:
    """Train and save a BPE tokenizer from a JSONL text field.

    JSONL rows are read one at a time and no temporary extracted training file
    is written. Optional ``im_start``/``im_end`` values are reserved as special
    tokens for later document-delimited LM use.
    """

    exe = Path(binary) if binary is not None else default_tokenizer_binary()
    args: list[str | Path] = [
        exe,
        "--tokenizer",
        tokenizer,
        "--input-jsonl",
        input_jsonl,
        "--jsonl-text-field",
        jsonl_text_field,
        "--train-only",
    ]
    _append_optional_im_tokens(args, im_start, im_end)
    _append_training_options(
        args,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        seed=seed,
    )
    return _run(args)


def tokenize_file_packed(
    *,
    tokenizer: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    im_start: str,
    im_end: str,
    num_tokens_per_sequence: int,
    max_length: int | None = None,
    use_tokenizer: bool = False,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = (),
    seed: int = 17,
    binary: str | Path | None = None,
) -> Mapping[str, object]:
    """Run the C tokenizer in single-text-file packed mode.

    By default ``tokenizer`` is the path where a newly trained BPE tokenizer is
    saved.  Set ``use_tokenizer=True`` to load and reuse an existing tokenizer
    from that path instead.

    The C tool builds one logical sequence as
    ``[im_start_id, body..., im_end_id]`` and slices that stream into fixed-size
    rows with one-token overlap.  For ``num_tokens_per_sequence=512`` the row
    stride is 511, so ``x=row[:-1]`` and ``y=row[1:]`` preserve the transition
    into the next row.  Continuation rows do not get synthetic start/end tokens.
    Body tokens that cannot fit while preserving a final ``im_end_id`` are
    dropped.
    """

    if num_tokens_per_sequence < 3:
        raise ValueError("num_tokens_per_sequence must be >= 3")
    exe = Path(binary) if binary is not None else default_tokenizer_binary()
    args: list[str | Path] = [
        exe,
        "--use-tokenizer" if use_tokenizer else "--tokenizer",
        tokenizer,
        "--input",
        input_path,
        "--output-dir",
        output_dir,
        "--im-start",
        im_start,
        "--im-end",
        im_end,
        "--num-tokens-per-sequence",
        str(num_tokens_per_sequence),
    ]
    if max_length is not None:
        args += ["--max-length", str(max_length)]
    if not use_tokenizer:
        _append_training_options(
            args,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            seed=seed,
        )
    return _run(args)


def tokenize_jsonl(
    *,
    tokenizer: str | Path,
    input_jsonl: str | Path,
    output_dir: str | Path,
    im_start: str,
    im_end: str,
    max_length: int,
    jsonl_text_field: str = "text",
    use_tokenizer: bool = False,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    special_tokens: Sequence[str] = (),
    seed: int = 17,
    binary: str | Path | None = None,
) -> Mapping[str, object]:
    """Run the C tokenizer in JSONL one-sequence-per-line mode.

    By default ``tokenizer`` is trained from the JSONL text field and saved.
    Training is streamed row-by-row without writing a temporary extracted text
    file. Set ``use_tokenizer=True`` to reuse an existing tokenizer from that
    path.
    """

    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    exe = Path(binary) if binary is not None else default_tokenizer_binary()
    args: list[str | Path] = [
        exe,
        "--use-tokenizer" if use_tokenizer else "--tokenizer",
        tokenizer,
        "--input-jsonl",
        input_jsonl,
        "--jsonl-text-field",
        jsonl_text_field,
        "--output-dir",
        output_dir,
        "--im-start",
        im_start,
        "--im-end",
        im_end,
        "--max-length",
        str(max_length),
    ]
    if not use_tokenizer:
        _append_training_options(
            args,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            seed=seed,
        )
    return _run(args)


@dataclass(frozen=True)
class TokenizedShard:
    """One shard entry from a tokenized corpus manifest."""

    directory: Path
    info: Mapping[str, object]

    @property
    def tokens_path(self) -> Path:
        return self.directory / str(self.info["tokens"])

    @property
    def offsets_path(self) -> Path | None:
        value = self.info.get("offsets")
        if value is None:
            return None
        return self.directory / str(value)

    @property
    def sequences(self) -> int:
        return int(self.info["sequences"])

    @property
    def tokens_count(self) -> int:
        return int(self.info["tokens_count"])


@dataclass(frozen=True)
class TokenizedCorpus:
    """Mmap-backed view of a tokenized corpus directory."""

    directory: Path
    manifest: Mapping[str, object]

    @classmethod
    def open(cls, path: str | Path) -> "TokenizedCorpus":
        directory = Path(path)
        manifest_path = directory / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != TOKENIZED_CORPUS_FORMAT:
            raise ValueError(f"unsupported tokenized corpus format: {manifest.get('format')!r}")
        return cls(directory=directory, manifest=manifest)

    @property
    def mode(self) -> str:
        return str(self.manifest["mode"])

    @property
    def num_sequences(self) -> int:
        return int(self.manifest["num_sequences"])

    @property
    def num_tokens(self) -> int:
        return int(self.manifest["num_tokens"])

    @property
    def sequence_length(self) -> int | None:
        value = int(self.manifest.get("sequence_length", 0))
        return value if value > 0 else None

    @property
    def max_length(self) -> int:
        return int(self.manifest["max_length"])

    @property
    def stride(self) -> int | None:
        value = self.manifest.get("stride")
        return None if value is None else int(value)

    @property
    def im_start_id(self) -> int:
        return int(self.manifest["im_start_id"])

    @property
    def im_end_id(self) -> int:
        return int(self.manifest["im_end_id"])

    @property
    def shards(self) -> list[TokenizedShard]:
        return [TokenizedShard(self.directory, s) for s in self.manifest.get("shards", [])]

    def mmap_packed_shard(self, shard: TokenizedShard) -> np.memmap:
        """Return one packed shard as ``[num_sequences, sequence_length]``."""

        if self.mode != "packed":
            raise ValueError(f"mmap_packed_shard requires packed mode, got {self.mode!r}")
        seq_len = self.sequence_length
        if seq_len is None:
            raise ValueError("packed corpus is missing sequence_length")
        arr = np.memmap(shard.tokens_path, dtype=TOKEN_DTYPE, mode="r")
        expected = shard.sequences * seq_len
        if arr.size != expected:
            raise ValueError(
                f"packed shard size mismatch for {shard.tokens_path}: "
                f"got {arr.size} tokens, expected {expected}"
            )
        return arr.reshape((shard.sequences, seq_len))

    def mmap_jsonl_shard(self, shard: TokenizedShard) -> tuple[np.memmap, np.memmap]:
        """Return ``(tokens, offsets)`` memmaps for one JSONL shard."""

        if self.mode != "jsonl":
            raise ValueError(f"mmap_jsonl_shard requires jsonl mode, got {self.mode!r}")
        offsets_path = shard.offsets_path
        if offsets_path is None:
            raise ValueError(f"jsonl shard {shard.tokens_path} has no offsets file")
        tokens = np.memmap(shard.tokens_path, dtype=TOKEN_DTYPE, mode="r")
        offsets = np.memmap(offsets_path, dtype=OFFSET_DTYPE, mode="r")
        if offsets.size != shard.sequences + 1:
            raise ValueError(
                f"offset count mismatch for {offsets_path}: "
                f"got {offsets.size}, expected {shard.sequences + 1}"
            )
        if int(offsets[0]) != 0 or int(offsets[-1]) != tokens.size:
            raise ValueError(
                f"invalid offsets for {offsets_path}: first={int(offsets[0])}, "
                f"last={int(offsets[-1])}, tokens={tokens.size}"
            )
        return tokens, offsets

    def iter_packed_arrays(self) -> Iterator[np.memmap]:
        """Yield packed shard arrays without copying."""

        for shard in self.shards:
            yield self.mmap_packed_shard(shard)

    def iter_sequences(self, *, copy: bool = False) -> Iterator[np.ndarray]:
        """Yield individual token sequences.

        In packed mode, yielded arrays are fixed-length rows.  In JSONL mode,
        yielded arrays are variable-length slices determined by shard offsets.
        Set ``copy=True`` when the caller needs arrays independent of the mmap.
        """

        if self.mode == "packed":
            for arr in self.iter_packed_arrays():
                for i in range(arr.shape[0]):
                    row = arr[i]
                    yield np.array(row, copy=True) if copy else row
            return
        if self.mode == "jsonl":
            for shard in self.shards:
                tokens, offsets = self.mmap_jsonl_shard(shard)
                for i in range(shard.sequences):
                    start = int(offsets[i])
                    end = int(offsets[i + 1])
                    seq = tokens[start:end]
                    yield np.array(seq, copy=True) if copy else seq
            return
        raise ValueError(f"unsupported tokenized corpus mode: {self.mode!r}")


__all__ = [
    "OFFSET_DTYPE",
    "TOKENIZED_CORPUS_FORMAT",
    "TOKEN_DTYPE",
    "TokenizerToolError",
    "TokenizedCorpus",
    "TokenizedShard",
    "default_tokenizer_binary",
    "tokenize_file_packed",
    "tokenize_jsonl",
    "train_tokenizer_file",
    "train_tokenizer_jsonl",
]
