#!/usr/bin/env bash
set -euo pipefail

# Full ImageNet GPT-VLM data build for gradients.c.
# Run from repo root or directly:
#   bash examples/gpt-vlm/prep-full-data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IN_DIR="/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched"
OUT_DIR="/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-gradients-224-16patch"
WORKERS="8"
SAMPLES_PER_SHARD="2048"
VOCAB_SIZE="2048"
BPE_MIN_FREQUENCY="1"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

printf '[gpt-vlm prep] repo=%s\n' "$REPO_ROOT"
printf '[gpt-vlm prep] in=%s\n' "$IN_DIR"
printf '[gpt-vlm prep] out=%s\n' "$OUT_DIR"
printf '[gpt-vlm prep] workers=%s samples_per_shard=%s vocab=%s\n' \
  "$WORKERS" "$SAMPLES_PER_SHARD" "$VOCAB_SIZE"

printf '\n[1/5] extract label text\n'
uv run "$SCRIPT_DIR/extract_label_text.py" \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR"

printf '\n[2/5] build gradients.c tools\n'
make tools

printf '\n[3/5] train tokenizer\n'
build/tools/gradients-train-bpe \
  --input "$OUT_DIR/label_text.txt" \
  --output "$OUT_DIR/tokenizer.json" \
  --vocab-size "$VOCAB_SIZE" \
  --min-frequency "$BPE_MIN_FREQUENCY" \
  --split-digits \
  --special '<|pad|>' \
  --special '<|im_start|>' \
  --special '<|im_end|>'

printf '\n[4/5] tokenize label text\n'
uv run "$SCRIPT_DIR/tokenize_label_text.py" \
  --tokenizer "$OUT_DIR/tokenizer.json" \
  --text "$OUT_DIR/label_text.txt" \
  --labels "$OUT_DIR/labels.txt" \
  --output "$OUT_DIR/text-tokenized.json"

printf '\n[5/5] preprocess train split\n'
uv run "$SCRIPT_DIR/prep_imagenet_vlm.py" \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --split train \
  --tokenized-text "$OUT_DIR/text-tokenized.json" \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  --num-workers "$WORKERS"

printf '\n[5/5] preprocess validation split\n'
uv run "$SCRIPT_DIR/prep_imagenet_vlm.py" \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --split validation \
  --tokenized-text "$OUT_DIR/text-tokenized.json" \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  --num-workers "$WORKERS"

printf '\n[gpt-vlm prep] done: %s\n' "$OUT_DIR"
