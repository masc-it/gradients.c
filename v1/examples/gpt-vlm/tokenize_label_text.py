#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Tokenize ImageNet VLM label_text.txt with gradients.c tokenizer.json.

This is a small Python implementation of gradients.c's gd-bpe-tokenizer-v1
encode path. Keeping tokenization in JSON form makes final image preprocessing
pure image work: it only looks up label_id -> token ids.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_OUT_DIR = (
    "/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-"
    "gradients-224-16patch"
)
DEFAULT_IM_START = "<|im_start|>"
DEFAULT_IM_END = "<|im_end|>"
MAX_PIECE_BYTES = 256


@dataclass(frozen=True)
class SpecialToken:
    text: str
    token_id: int
    data: bytes


class GradientsBPETokenizer:
    """Exact-enough encoder for gradients.c gd-bpe-tokenizer-v1.

    The C tokenizer is byte-level BPE. Byte token ids are 0..255, special tokens
    are matched by literal byte string, and BPE merge priority is token id/rank.
    """

    def __init__(self, tokenizer_json: dict):
        fmt = tokenizer_json.get("format")
        if fmt != "gd-bpe-tokenizer-v1":
            raise ValueError(f"unsupported tokenizer format: {fmt!r}")
        self.split_digits = bool(tokenizer_json.get("split_digits", True))
        self.vocab_size = int(tokenizer_json["vocab_size"])
        self.hash = str(tokenizer_json.get("hash", ""))
        self.pair_to_id: dict[tuple[int, int], int] = {}
        self.byte_to_id: dict[int, int] = {}
        specials: list[SpecialToken] = []

        tokens = tokenizer_json.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("tokenizer JSON missing tokens list")
        for obj in tokens:
            token_id = int(obj["id"])
            kind = obj["kind"]
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token id out of range: {token_id}")
            if kind == "byte":
                b = int(obj["hex"], 16)
                self.byte_to_id[b] = token_id
            elif kind == "merge":
                self.pair_to_id[(int(obj["left"]), int(obj["right"]))] = token_id
            elif kind == "special":
                text = str(obj["text"])
                specials.append(SpecialToken(text=text, token_id=token_id, data=text.encode("utf-8")))
            else:
                raise ValueError(f"unknown token kind: {kind!r}")

        missing = [b for b in range(256) if self.byte_to_id.get(b) != b]
        if missing:
            raise ValueError("gradients.c tokenizer requires byte token ids 0..255")
        self.specials = specials
        self.special_id_by_text = {s.text: s.token_id for s in specials}

    @classmethod
    def load(cls, path: Path) -> "GradientsBPETokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _is_digit(c: int) -> bool:
        return ord("0") <= c <= ord("9")

    @staticmethod
    def _is_space(c: int) -> bool:
        return c in (ord(" "), ord("\n"), ord("\r"), ord("\t"), 0x0B, 0x0C)

    @staticmethod
    def _is_alpha(c: int) -> bool:
        return (ord("A") <= c <= ord("Z")) or (ord("a") <= c <= ord("z")) or c >= 0x80

    @classmethod
    def _classify(cls, c: int) -> int:
        if cls._is_space(c):
            return 0
        if cls._is_digit(c):
            return 1
        if cls._is_alpha(c):
            return 2
        return 3

    def _match_special(self, data: bytes, pos: int) -> tuple[int, int | None]:
        best_len = 0
        best_id: int | None = None
        for special in self.specials:
            n = len(special.data)
            if n > best_len and data.startswith(special.data, pos):
                best_len = n
                best_id = special.token_id
        return best_len, best_id

    def _encode_piece(self, piece: bytes) -> list[int]:
        if not piece:
            return []
        if len(piece) > MAX_PIECE_BYTES:
            raise ValueError("internal error: BPE piece exceeds max chunk size")
        parts = [self.byte_to_id[b] for b in piece]
        while len(parts) > 1:
            best_index = -1
            best_rank = 2**31 - 1
            for i in range(len(parts) - 1):
                rank = self.pair_to_id.get((parts[i], parts[i + 1]), -1)
                if 0 <= rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            parts[best_index] = best_rank
            del parts[best_index + 1]
        return parts

    def _emit_normal_piece(self, piece: bytes) -> list[int]:
        ids: list[int] = []
        for offset in range(0, len(piece), MAX_PIECE_BYTES):
            ids.extend(self._encode_piece(piece[offset : offset + MAX_PIECE_BYTES]))
        return ids

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        ids: list[int] = []
        i = 0
        n = len(data)
        while i < n:
            match_len, special_id = self._match_special(data, i)
            if match_len > 0:
                assert special_id is not None
                ids.append(special_id)
                i += match_len
                continue

            if self.split_digits and self._is_digit(data[i]):
                ids.extend(self._emit_normal_piece(data[i : i + 1]))
                i += 1
                continue

            if self._is_space(data[i]):
                start = i
                if (
                    data[i] == ord(" ")
                    and i + 1 < n
                    and not self._is_space(data[i + 1])
                    and not (self.split_digits and self._is_digit(data[i + 1]))
                    and self._match_special(data, i + 1)[0] == 0
                ):
                    i += 1
                    cls = self._classify(data[i])
                    while (
                        i < n
                        and self._classify(data[i]) == cls
                        and not (self.split_digits and self._is_digit(data[i]))
                        and self._match_special(data, i)[0] == 0
                    ):
                        i += 1
                else:
                    i += 1
                    while i < n and self._is_space(data[i]) and self._match_special(data, i)[0] == 0:
                        i += 1
                ids.extend(self._emit_normal_piece(data[start:i]))
                continue

            start = i
            cls = self._classify(data[i])
            i += 1
            while (
                i < n
                and self._classify(data[i]) == cls
                and not (self.split_digits and self._is_digit(data[i]))
                and self._match_special(data, i)[0] == 0
            ):
                i += 1
            ids.extend(self._emit_normal_piece(data[start:i]))
        return ids


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def infer_label(text: str, im_start: str, im_end: str) -> str:
    if text.startswith(im_start) and text.endswith(im_end):
        return text[len(im_start) : len(text) - len(im_end)]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize ImageNet VLM label text")
    parser.add_argument("--tokenizer", default=str(Path(DEFAULT_OUT_DIR) / "tokenizer.json"))
    parser.add_argument("--text", default=str(Path(DEFAULT_OUT_DIR) / "label_text.txt"))
    parser.add_argument("--labels", default=str(Path(DEFAULT_OUT_DIR) / "labels.txt"))
    parser.add_argument("--output", default=str(Path(DEFAULT_OUT_DIR) / "text-tokenized.json"))
    parser.add_argument("--im-start", default=DEFAULT_IM_START)
    parser.add_argument("--im-end", default=DEFAULT_IM_END)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    text_path = Path(args.text)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    tok = GradientsBPETokenizer.load(tokenizer_path)
    texts = read_lines(text_path)
    labels = read_lines(labels_path) if labels_path.is_file() else [infer_label(t, args.im_start, args.im_end) for t in texts]
    if len(labels) != len(texts):
        raise ValueError(f"labels/text line count mismatch: labels={len(labels)} text={len(texts)}")

    for special in ("<|pad|>", args.im_start, args.im_end):
        if special not in tok.special_id_by_text:
            raise ValueError(f"tokenizer missing required special token: {special}")

    entries = []
    lengths = []
    for label_id, (label, text) in enumerate(zip(labels, texts, strict=True)):
        expected = f"{args.im_start}{label}{args.im_end}"
        if text != expected:
            raise ValueError(
                f"line {label_id} text does not match expected format: got {text!r}, expected {expected!r}"
            )
        ids = tok.encode(text)
        if not ids:
            raise ValueError(f"line {label_id} encoded to zero tokens")
        if any(i < 0 or i >= tok.vocab_size for i in ids):
            raise ValueError(f"line {label_id} has token id outside vocab")
        lengths.append(len(ids))
        entries.append(
            {
                "label_id": label_id,
                "label": label,
                "text": text,
                "ids": ids,
                "n_tokens": len(ids),
            }
        )

    payload = {
        "format": "gd-imagenet-vlm-text-tokenized-v1",
        "text_format": f"{args.im_start}label{args.im_end}",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tok.hash,
        "vocab_size": tok.vocab_size,
        "split_digits": tok.split_digits,
        "pad_id": tok.special_id_by_text["<|pad|>"],
        "im_start_id": tok.special_id_by_text[args.im_start],
        "im_end_id": tok.special_id_by_text[args.im_end],
        "labels": entries,
        "stats": {
            "n_labels": len(entries),
            "total_text_tokens_one_each": sum(lengths),
            "min_text_tokens": min(lengths),
            "max_text_tokens": max(lengths),
            "mean_text_tokens": mean(lengths),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)

    print(f"tokenizer: {tokenizer_path}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"hash: {tok.hash}")
    print(f"wrote: {output_path}")
    print(
        "label text tokens: "
        f"labels={len(entries)} total={sum(lengths)} "
        f"min={min(lengths)} max={max(lengths)} mean={mean(lengths):.2f}"
    )


if __name__ == "__main__":
    main()
