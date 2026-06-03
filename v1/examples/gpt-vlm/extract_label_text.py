#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "pyarrow",
#     "pyyaml",
# ]
# ///
"""Extract ImageNet class-label text for gradients.c GPT-VLM preprocessing.

Writes two files into --out-dir:
  labels.txt       one label name per line; line index == ImageNet label id
  label_text.txt   one generation target per line:
                   <|im_start|>label<|im_end|>

The tokenizer must be trained on label_text.txt before image preprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_IN_DIR = "/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched"
DEFAULT_OUT_DIR = (
    "/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-"
    "gradients-224-16patch"
)
DEFAULT_IM_START = "<|im_start|>"
DEFAULT_IM_END = "<|im_end|>"


def _parquet_files(in_dir: Path, split: str) -> list[Path]:
    data_dir = in_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing parquet data dir: {data_dir}")
    files = sorted(data_dir.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards for split '{split}' in {data_dir}")
    return files


def _labels_from_datasets(first_parquet: Path) -> list[str]:
    # Matches dnn.c reference script. It asks HuggingFace datasets for the
    # ClassLabel feature attached to the parquet metadata.
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(first_parquet), split="train", streaming=False)
    label_feature = ds.features["label"]
    names = getattr(label_feature, "names", None)
    if not names:
        raise ValueError("datasets feature 'label' has no ClassLabel names")
    return [str(x) for x in names]


def _labels_from_readme(in_dir: Path) -> list[str]:
    # Fallback for local clones where datasets cannot recover ClassLabel names.
    # The enriched dataset README has a YAML frontmatter dataset_info block.
    import yaml

    readme = in_dir / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"missing README fallback: {readme}")
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("README has no YAML frontmatter")
    try:
        _, frontmatter, _rest = text.split("---", 2)
    except ValueError as exc:
        raise ValueError("README frontmatter is not delimited by ---") from exc
    meta = yaml.safe_load(frontmatter)
    features = meta.get("dataset_info", {}).get("features", [])
    for feat in features:
        if feat.get("name") != "label":
            continue
        names = feat.get("dtype", {}).get("class_label", {}).get("names")
        if isinstance(names, dict):
            items = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda kv: kv[0])
            return [v for _, v in items]
        if isinstance(names, list):
            return [str(v) for v in names]
    raise ValueError("README dataset_info lacks label class names")


def load_label_names(in_dir: Path, split: str) -> list[str]:
    first = _parquet_files(in_dir, split)[0]
    try:
        labels = _labels_from_datasets(first)
        source = "datasets"
    except Exception as exc:  # pragma: no cover - fallback path is environment-specific.
        print(f"warning: datasets label extraction failed: {exc}")
        labels = _labels_from_readme(in_dir)
        source = "README.md"
    if not labels:
        raise ValueError("empty label list")
    for i, label in enumerate(labels):
        if not label:
            raise ValueError(f"empty label name at id {i}")
        if "\n" in label or "\r" in label:
            raise ValueError(f"label contains newline at id {i}: {label!r}")
    print(f"labels: loaded {len(labels)} names from {source}")
    return labels


def write_lines(path: Path, lines: Iterable[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ImageNet VLM label text")
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--im-start", default=DEFAULT_IM_START)
    parser.add_argument("--im-end", default=DEFAULT_IM_END)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label_names(in_dir, args.split)
    texts = [f"{args.im_start}{label}{args.im_end}" for label in labels]

    labels_path = out_dir / "labels.txt"
    text_path = out_dir / "label_text.txt"
    meta_path = out_dir / "label_text.meta.json"

    write_lines(labels_path, labels)
    write_lines(text_path, texts)

    meta = {
        "format": "gd-imagenet-vlm-label-text-v1",
        "source_dir": str(in_dir),
        "split_used_for_labels": args.split,
        "n_labels": len(labels),
        "im_start": args.im_start,
        "im_end": args.im_end,
        "labels_path": str(labels_path),
        "label_text_path": str(text_path),
        "text_format": f"{args.im_start}label{args.im_end}",
    }
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    total_chars = sum(len(t) for t in texts)
    print(f"wrote: {labels_path}")
    print(f"wrote: {text_path}")
    print(f"wrote: {meta_path}")
    print(f"label text chars: total={total_chars} avg={total_chars / len(texts):.2f}")


if __name__ == "__main__":
    main()
