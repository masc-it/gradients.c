# GPT-VLM ImageNet data prep

Builds gradients.c-ready ImageNet VLM data:

```text
[196 image patch tokens] + [text tokens]
text = <|im_start|>label<|im_end|>
```

Image preprocessing:
- resize short edge to `256`, center-crop `224x224`
- ImageNet mean/std normalize
- split into `14x14` patches of `16x16x3`
- store patch matrix `[196, 768]` as little-endian `fp16`

Pipeline:

1. `extract_label_text.py`
   - writes `labels.txt`
   - writes `label_text.txt`

2. `gradients-train-bpe`
   - trains gradients.c BPE tokenizer on `label_text.txt`
   - vocab size `2048`
   - specials: `<|pad|>`, `<|im_start|>`, `<|im_end|>`

3. `tokenize_label_text.py`
   - reads `tokenizer.json` + `label_text.txt`
   - writes `text-tokenized.json`

4. `prep_imagenet_vlm.py`
   - reads parquet images + `text-tokenized.json`
   - writes `*.gdvlm` shards, per-shard `*.idx`, combined split idx, split meta
   - prints complete split image/text/total token counts

Default output dir:

```text
/Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-gradients-224-16patch
```

Shard record:

```text
int32  label_id
uint32 token_len
u16/u32 token_ids[token_len]
fp16   patches[196][768]
```

Loss convention for loader/training:
- image prefix tokens: `196`
- causal attention with `prefix_len=196`
- loss mask ignores image prefix; text suffix only
