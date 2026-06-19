# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=0.23",
#   "numpy>=1.26",
#   "safetensors>=0.4",
#   "torch>=2.2",
#   "transformers>=4.51",
# ]
# ///
"""Initialize examples/gpt_lm from Qwen3 weights.

This is a *checkpoint compiler*: it leaves the C target architecture unchanged and
writes a gradients.c gdckpt state dict whose tensor paths/shapes match gpt_lm.

Example, from repo root:

    uv run examples/gpt_lm/qwen3_init.py \
      --repo-id Qwen/Qwen3-0.6B \
      --target-tokenizer examples/gpt_lm/data/tokenizer-v2048.json \
      --layers 3 \
      --projection pca \
      --depth-blend gaussian \
      --rope-map frequency \
      --mlp-select energy \
      --out examples/gpt_lm/checkpoints/gpt_lm_qwen3_0p6B_spectral_l3.gdckpt

Then train/generate with:

    make -C examples/gpt_lm run ARGS="--layers 3 --load-checkpoint checkpoints/gpt_lm_qwen3_0p6B_spectral_l3.gdckpt --epochs 1 --batch-size 32 --lr-max 1e-4"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


GD_CKPT_MAGIC = b"GDCKPT1\0"
GD_CKPT_VERSION = 1
GD_CKPT_HEADER_SIZE = 64
GD_CKPT_ENTRY_PARAM = 1
GD_DTYPE_F16 = 1
GD_MAX_DIMS = 8

TARGET_VOCAB = 2048
TARGET_CONTEXT = 512
TARGET_D_MODEL = 256
TARGET_HEADS = 4
TARGET_HEAD_DIM = 64
TARGET_MLP = 1024
TARGET_SDPA_WINDOW = 256
TARGET_WEIGHT_STD = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="Qwen/Qwen3-0.6B", help="HF repo id")
    p.add_argument("--revision", default=None, help="HF revision/commit")
    p.add_argument("--model-dir", default=None, help="Use an existing snapshot directory instead of downloading")
    p.add_argument("--target-tokenizer", default="examples/gpt_lm/data/tokenizer-v2048.json")
    p.add_argument("--out", default="examples/gpt_lm/checkpoints/gpt_lm_qwen3_init.gdckpt")
    p.add_argument("--layers", type=int, default=7, help="Target gpt_lm layer count")
    p.add_argument("--projection", choices=("random", "pca"), default="pca")
    p.add_argument("--pca-rows", type=int, default=32768, help="Source embedding rows sampled for PCA")
    p.add_argument("--pca-niter", type=int, default=2)
    p.add_argument("--layer-map", choices=("linear", "center"), default="linear")
    p.add_argument("--depth-blend", choices=("pick", "gaussian"), default="pick",
                   help="pick one source layer or fold nearby source layers with Gaussian weights")
    p.add_argument("--blend-width", type=float, default=0.55,
                   help="Gaussian sigma as a fraction of target block spacing")
    p.add_argument("--blend-topk", type=int, default=7,
                   help="For Gaussian depth folding, keep only nearest K source layers per target block")
    p.add_argument("--rope-map", choices=("linear", "frequency"), default="linear",
                   help="Map Q/K RoPE channels linearly or by matching source/target RoPE frequencies")
    p.add_argument("--mlp-select", choices=("linear", "energy"), default="linear",
                   help="Select source MLP neurons uniformly or by gate/up/down path norm")
    p.add_argument("--seed", type=int, default=0x5179_9EED)
    p.add_argument("--no-rescale", action="store_true", help="Do not RMS-rescale transplanted tensors")
    p.add_argument("--embedding-only", action="store_true", help="Only transplant token embedding/LM head; random-init blocks")
    return p.parse_args()


@dataclass
class TargetEntry:
    path: str
    array: np.ndarray  # np.float16, C-contiguous


class TensorStore:
    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = model_dir
        self.weight_map: dict[str, str] = {}
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            self.weight_map = dict(data["weight_map"])
        else:
            for sf in sorted(model_dir.glob("*.safetensors")):
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = sf.name
        if not self.weight_map:
            raise FileNotFoundError(f"no safetensors weights found in {model_dir}")

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"tensor not found in HF checkpoint: {name}")
        path = self.model_dir / self.weight_map[name]
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(torch.float32)


def resolve_model_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.model_dir:
        return pathlib.Path(args.model_dir).expanduser().resolve()
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    return pathlib.Path(path).resolve()


def random_orthogonal_projection(d_src: int, d_tgt: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d_src, d_tgt, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.contiguous()


def pca_projection(src_embed: torch.Tensor, d_tgt: int, rows: int, niter: int, seed: int) -> torch.Tensor:
    n = min(rows, src_embed.shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(src_embed.shape[0], generator=g)[:n]
    x = src_embed[idx].to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    # V is [hidden, q]; columns are orthonormal principal directions.
    _, _, v = torch.pca_lowrank(x, q=d_tgt, center=False, niter=niter)
    return v[:, :d_tgt].contiguous()


def target_token_to_text(item: dict) -> str | None:
    if item.get("kind") == "special":
        return item.get("text")
    hx = item.get("hex")
    if not hx:
        return None
    raw = bytes.fromhex(hx)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # The target tokenizer is byte-level. Most non-UTF8 singleton bytes have
        # no clean text equivalent in Qwen's tokenizer; leave them random.
        if len(raw) == 1 and raw[0] < 128:
            return raw.decode("latin-1")
        return None


def load_target_token_texts(path: pathlib.Path) -> list[str | None]:
    data = json.loads(path.read_text())
    toks = sorted(data["tokens"], key=lambda x: x["id"])
    if len(toks) != TARGET_VOCAB:
        raise ValueError(f"expected {TARGET_VOCAB} target tokens, found {len(toks)}")
    return [target_token_to_text(t) for t in toks]


def qwen_ids_for_text(tokenizer, text: str | None) -> list[int]:
    if text is None:
        return []
    if text.startswith("<|") and text.endswith("|>"):
        sid = tokenizer.convert_tokens_to_ids(text)
        if isinstance(sid, int) and sid >= 0:
            return [sid]
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return [int(x) for x in ids if int(x) >= 0]


def normalize_std(t: torch.Tensor, desired_std: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return t
    cur = t.float().std(unbiased=False).item()
    if cur > 1.0e-12 and math.isfinite(cur):
        t = t * (desired_std / cur)
    return t


def torch_to_f16_np(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().to(torch.float16).numpy()
    return np.ascontiguousarray(arr.astype("<f2", copy=False))


def np_to_f16_np(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.astype("<f2", copy=False))


def random_f16(shape: tuple[int, ...], std: float, rng: np.random.Generator) -> np.ndarray:
    return np_to_f16_np(rng.normal(0.0, std, size=shape).astype(np.float32))


def compose_target_embedding(
    src_embed: torch.Tensor,
    projection: torch.Tensor,
    qwen_tokenizer,
    target_tokenizer_path: pathlib.Path,
    rng: np.random.Generator,
    rescale: bool,
) -> tuple[np.ndarray, int]:
    token_texts = load_target_token_texts(target_tokenizer_path)
    rows: list[torch.Tensor] = []
    mapped = 0
    for text in token_texts:
        ids = qwen_ids_for_text(qwen_tokenizer, text)
        if ids:
            e = src_embed[torch.tensor(ids, dtype=torch.long)].mean(dim=0)
            row = e @ projection
            rows.append(row)
            mapped += 1
        else:
            fallback = rng.normal(0.0, TARGET_WEIGHT_STD, size=(TARGET_D_MODEL,)).astype(np.float32)
            rows.append(torch.from_numpy(fallback))
    emb = torch.stack(rows, dim=0).to(torch.float32)
    emb = normalize_std(emb, TARGET_WEIGHT_STD, rescale)
    return torch_to_f16_np(emb), mapped


def layer_index(j: int, target_layers: int, source_layers: int, mode: str) -> int:
    if target_layers <= 1:
        return source_layers // 2
    if mode == "center":
        return min(source_layers - 1, int((j + 0.5) * source_layers / target_layers))
    return int(round(j * (source_layers - 1) / (target_layers - 1)))


def source_layer_weights(j: int, target_layers: int, source_layers: int, args: argparse.Namespace) -> list[tuple[int, float]]:
    """Map one target block to source blocks.

    The Gaussian mode is a cheap form of depth distillation: instead of taking
    only source layers {0, mid, last}, each tiny target block gets a local
    weighted "layer soup" from the teacher depth.
    """
    center = float(layer_index(j, target_layers, source_layers, args.layer_map))
    if args.depth_blend == "pick":
        return [(int(round(center)), 1.0)]
    if target_layers <= 1:
        spacing = max(1.0, float(source_layers))
    else:
        spacing = max(1.0, float(source_layers - 1) / float(target_layers - 1))
    sigma = max(1.0e-6, args.blend_width * spacing)
    all_weights = []
    for i in range(source_layers):
        w = math.exp(-0.5 * ((float(i) - center) / sigma) ** 2)
        all_weights.append((i, w))
    topk = min(max(1, int(args.blend_topk)), source_layers)
    chosen = sorted(sorted(all_weights, key=lambda x: x[1], reverse=True)[:topk], key=lambda x: x[0])
    total = sum(w for _, w in chosen)
    return [(i, w / total) for i, w in chosen]


def source_rope_theta(cfg) -> float:
    params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    theta = params.get("rope_theta") if isinstance(params, dict) else None
    if theta is None:
        theta = getattr(cfg, "rope_theta", None)
    return float(theta if theta is not None else 10000.0)


def neox_rope_dim_indices(src_head_dim: int, target_head_dim: int, cfg) -> torch.Tensor:
    # gpt_lm uses interleaved=false RoPE: pair i with i + head_dim/2.
    src_pairs = src_head_dim // 2
    tgt_pairs = target_head_dim // 2
    if args_global.rope_map == "frequency":
        # Qwen3 uses theta=1e6 while this GPT uses theta=1e4.  Match channels by
        # actual RoPE frequency, not by raw channel index, so copied Q/K weights
        # land in target positions that rotate at similar rates.
        src_theta = source_rope_theta(cfg)
        tgt_theta = 10000.0
        src_log_freq = -2.0 * torch.arange(src_pairs, dtype=torch.float32) / float(src_head_dim) * math.log(src_theta)
        tgt_log_freq = -2.0 * torch.arange(tgt_pairs, dtype=torch.float32) / float(target_head_dim) * math.log(tgt_theta)
        freq = torch.argmin(torch.abs(src_log_freq[:, None] - tgt_log_freq[None, :]), dim=0).to(torch.long)
    else:
        freq = torch.linspace(0, src_pairs - 1, tgt_pairs).round().to(torch.long)
    return torch.cat([freq, freq + src_pairs], dim=0)


def source_head_ids(source_heads: int, target_heads: int) -> list[int]:
    if target_heads == 1:
        return [source_heads // 2]
    return [min(source_heads - 1, int(round(h * (source_heads - 1) / (target_heads - 1)))) for h in range(target_heads)]


def build_qkv(store: TensorStore, cfg, src_layer: int, projection: torch.Tensor, rescale: bool) -> torch.Tensor:
    prefix = f"model.layers.{src_layer}.self_attn"
    src_heads = int(cfg.num_attention_heads)
    src_kv_heads = int(getattr(cfg, "num_key_value_heads", src_heads))
    src_head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // src_heads))
    head_ids = source_head_ids(src_heads, TARGET_HEADS)
    kv_group = max(1, src_heads // src_kv_heads)
    kv_ids = [min(src_kv_heads - 1, h // kv_group) for h in head_ids]
    dim_idx = neox_rope_dim_indices(src_head_dim, TARGET_HEAD_DIM, cfg)

    def project_heads(name: str, n_heads: int, ids: Iterable[int]) -> torch.Tensor:
        w = store.get(f"{prefix}.{name}.weight")
        w = w.reshape(n_heads, src_head_dim, int(cfg.hidden_size))
        parts = []
        for hid in ids:
            parts.append(w[hid, dim_idx, :] @ projection)
        return torch.stack(parts, dim=0).reshape(TARGET_HEADS * TARGET_HEAD_DIM, TARGET_D_MODEL)

    q = project_heads("q_proj", src_heads, head_ids)
    k = project_heads("k_proj", src_kv_heads, kv_ids)
    v = project_heads("v_proj", src_kv_heads, kv_ids)
    qkv = torch.cat([q, k, v], dim=0)
    return normalize_std(qkv, TARGET_WEIGHT_STD, rescale)


def build_attn_out(store: TensorStore, cfg, src_layer: int, projection: torch.Tensor, rescale: bool) -> torch.Tensor:
    prefix = f"model.layers.{src_layer}.self_attn"
    src_heads = int(cfg.num_attention_heads)
    src_head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // src_heads))
    head_ids = source_head_ids(src_heads, TARGET_HEADS)
    dim_idx = neox_rope_dim_indices(src_head_dim, TARGET_HEAD_DIM, cfg)
    cols = []
    for hid in head_ids:
        base = hid * src_head_dim
        cols.extend((base + dim_idx).tolist())
    w = store.get(f"{prefix}.o_proj.weight")  # [hidden, hidden], PyTorch Linear weight
    out = projection.T @ w[:, torch.tensor(cols, dtype=torch.long)]
    residual_std = TARGET_WEIGHT_STD / math.sqrt(2.0 * float(args_global.layers))
    return normalize_std(out, residual_std, rescale)


def mlp_indices(store: TensorStore, cfg, src_layer: int) -> torch.Tensor:
    source_intermediate = int(cfg.intermediate_size)
    if args_global.mlp_select == "energy":
        # Cheap neuron coreset: keep teacher MLP channels with high full-path
        # capacity.  A useful neuron needs strong gate/up input weights and a
        # strong down-projection column.
        prefix = f"model.layers.{src_layer}.mlp"
        up_norm = store.get(f"{prefix}.up_proj.weight").norm(dim=1)
        gate_norm = store.get(f"{prefix}.gate_proj.weight").norm(dim=1)
        down_norm = store.get(f"{prefix}.down_proj.weight").norm(dim=0)
        score = torch.sqrt(torch.clamp(up_norm * gate_norm, min=0.0)) * down_norm
        return torch.topk(score, k=TARGET_MLP, largest=True).indices.sort().values.to(torch.long)
    return torch.linspace(0, source_intermediate - 1, TARGET_MLP).round().to(torch.long)


def build_up_gate(store: TensorStore, cfg, src_layer: int, projection: torch.Tensor, rescale: bool) -> torch.Tensor:
    prefix = f"model.layers.{src_layer}.mlp"
    idx = mlp_indices(store, cfg, src_layer)
    up = store.get(f"{prefix}.up_proj.weight")[idx, :] @ projection
    gate = store.get(f"{prefix}.gate_proj.weight")[idx, :] @ projection
    out = torch.cat([up, gate], dim=0)  # gd_swiglu_split computes x1 * silu(x2)
    return normalize_std(out, TARGET_WEIGHT_STD, rescale)


def build_down(store: TensorStore, cfg, src_layer: int, projection: torch.Tensor, rescale: bool) -> torch.Tensor:
    prefix = f"model.layers.{src_layer}.mlp"
    idx = mlp_indices(store, cfg, src_layer)
    # Qwen down_proj is PyTorch [hidden, intermediate]. gpt_lm down_proj is gd_linear
    # weight [intermediate, hidden], so transpose the selected source columns.
    w = store.get(f"{prefix}.down_proj.weight")
    out = w[:, idx].T @ projection
    residual_std = TARGET_WEIGHT_STD / math.sqrt(2.0 * float(args_global.layers))
    return normalize_std(out, residual_std, rescale)


def blend_tensors(weighted_layers: list[tuple[int, float]], build_one) -> torch.Tensor:
    acc = None
    for src_layer, weight in weighted_layers:
        t = build_one(src_layer)
        acc = t * float(weight) if acc is None else acc + t * float(weight)
    if acc is None:
        raise ValueError("empty layer blend")
    return acc


def format_layer_weights(weighted_layers: list[tuple[int, float]]) -> str:
    return ", ".join(f"{i}:{w:.2f}" for i, w in weighted_layers)


def make_metadata(args: argparse.Namespace, cfg, mapped_tokens: int) -> bytes:
    lines = [
        "model=gpt_lm",
        f"init_from={args.repo_id}",
        f"source_hidden_size={cfg.hidden_size}",
        f"source_layers={cfg.num_hidden_layers}",
        f"source_heads={cfg.num_attention_heads}",
        f"source_kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}",
        f"source_intermediate={cfg.intermediate_size}",
        f"projection={args.projection}",
        f"depth_blend={args.depth_blend}",
        f"blend_width={args.blend_width}",
        f"blend_topk={args.blend_topk}",
        f"rope_map={args.rope_map}",
        f"mlp_select={args.mlp_select}",
        f"mapped_target_tokens={mapped_tokens}/{TARGET_VOCAB}",
        f"vocab_size={TARGET_VOCAB}",
        f"context_length={TARGET_CONTEXT}",
        f"d_model={TARGET_D_MODEL}",
        f"n_layers={args.layers}",
        f"n_heads={TARGET_HEADS}",
        f"head_dim={TARGET_HEAD_DIM}",
        f"mlp_hidden={TARGET_MLP}",
        f"sdpa_window={TARGET_SDPA_WINDOW}",
        "dropout=0.100000001",
        "tokenizer_path=data/tokenizer-v2048.json",
        "note=Qwen3 weight-surgery init; token_embedding and untied lm_head start from same mapped weights; all biases start at zero; follow with KL/CE distillation",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_gdckpt(entries: list[TargetEntry], out_path: pathlib.Path, metadata: bytes) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dir_size = 0
    for e in entries:
        path_bytes = e.path.encode("utf-8")
        if not path_bytes or len(path_bytes) >= 4096:
            raise ValueError(f"bad checkpoint path {e.path!r}")
        if e.array.dtype != np.dtype("<f2") and e.array.dtype != np.float16:
            raise ValueError(f"{e.path}: expected f16 array, got {e.array.dtype}")
        dir_size += 16 + 8 * GD_MAX_DIMS + 16 + len(path_bytes)
    dir_off = GD_CKPT_HEADER_SIZE
    meta_off = dir_off + dir_size
    data_off = meta_off + len(metadata)
    cursor = data_off
    data_offsets = []
    for e in entries:
        nbytes = int(e.array.nbytes)
        data_offsets.append(cursor)
        cursor += nbytes

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(
            struct.pack(
                "<8sIIQQQQQQ",
                GD_CKPT_MAGIC,
                GD_CKPT_VERSION,
                GD_CKPT_HEADER_SIZE,
                len(entries),
                dir_off,
                dir_size,
                meta_off,
                len(metadata),
                data_off,
            )
        )
        for e, off in zip(entries, data_offsets):
            arr = np.ascontiguousarray(e.array.astype("<f2", copy=False))
            path_bytes = e.path.encode("utf-8")
            shape = list(arr.shape) + [0] * (GD_MAX_DIMS - arr.ndim)
            f.write(struct.pack("<IIII", len(path_bytes), GD_CKPT_ENTRY_PARAM, GD_DTYPE_F16, arr.ndim))
            f.write(struct.pack("<" + "Q" * GD_MAX_DIMS, *[int(x) for x in shape]))
            f.write(struct.pack("<QQ", off, arr.nbytes))
            f.write(path_bytes)
        f.write(metadata)
        for e in entries:
            arr = np.ascontiguousarray(e.array.astype("<f2", copy=False))
            f.write(arr.tobytes(order="C"))
    os.replace(tmp, out_path)


def build_entries(args: argparse.Namespace, model_dir: pathlib.Path) -> list[TargetEntry]:
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    qtok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    store = TensorStore(model_dir)
    rng = np.random.default_rng(args.seed)
    rescale = not args.no_rescale

    print(f"source: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)} intermediate={cfg.intermediate_size}")
    src_embed = store.get("model.embed_tokens.weight")
    if args.projection == "pca":
        print(f"projection: PCA over {min(args.pca_rows, src_embed.shape[0])} embedding rows")
        projection = pca_projection(src_embed, TARGET_D_MODEL, args.pca_rows, args.pca_niter, args.seed)
    else:
        print("projection: random orthogonal")
        projection = random_orthogonal_projection(int(cfg.hidden_size), TARGET_D_MODEL, args.seed)

    print("building target token embedding / untied LM head init")
    emb, mapped_tokens = compose_target_embedding(
        src_embed,
        projection,
        qtok,
        pathlib.Path(args.target_tokenizer),
        rng,
        rescale,
    )
    del src_embed

    entries: list[TargetEntry] = [
        TargetEntry("gpt_lm.token_embedding", emb),
        TargetEntry("gpt_lm.lm_head", emb.copy()),
        TargetEntry("gpt_lm.lm_head_bias", np_to_f16_np(np.zeros((TARGET_VOCAB,), dtype=np.float32))),
        TargetEntry("gpt_lm.final_norm_w", np_to_f16_np(np.ones((TARGET_D_MODEL,), dtype=np.float32))),
    ]

    for j in range(args.layers):
        weighted_layers = source_layer_weights(j, args.layers, int(cfg.num_hidden_layers), args)
        if args.depth_blend == "pick":
            print(f"block {j}: source layer {weighted_layers[0][0]}")
        else:
            print(f"block {j}: source blend {format_layer_weights(weighted_layers)}")
        b = f"gpt_lm.blocks.{j}"
        entries.append(TargetEntry(f"{b}.attn_norm_w", np_to_f16_np(np.ones((TARGET_D_MODEL,), dtype=np.float32))))
        entries.append(TargetEntry(f"{b}.mlp_norm_w", np_to_f16_np(np.ones((TARGET_D_MODEL,), dtype=np.float32))))
        residual_std = TARGET_WEIGHT_STD / math.sqrt(2.0 * float(args.layers))
        if args.embedding_only:
            entries.append(TargetEntry(f"{b}.qkv_proj.weight", random_f16((3 * TARGET_D_MODEL, TARGET_D_MODEL), TARGET_WEIGHT_STD, rng)))
            entries.append(TargetEntry(f"{b}.attn_proj.weight", random_f16((TARGET_D_MODEL, TARGET_D_MODEL), residual_std, rng)))
            entries.append(TargetEntry(f"{b}.up_gate.weight", random_f16((2 * TARGET_MLP, TARGET_D_MODEL), TARGET_WEIGHT_STD, rng)))
            entries.append(TargetEntry(f"{b}.down_proj.weight", random_f16((TARGET_MLP, TARGET_D_MODEL), residual_std, rng)))
        else:
            qkv = blend_tensors(weighted_layers, lambda src: build_qkv(store, cfg, src, projection, False))
            attn_out = blend_tensors(weighted_layers, lambda src: build_attn_out(store, cfg, src, projection, False))
            up_gate = blend_tensors(weighted_layers, lambda src: build_up_gate(store, cfg, src, projection, False))
            down = blend_tensors(weighted_layers, lambda src: build_down(store, cfg, src, projection, False))
            entries.append(TargetEntry(f"{b}.qkv_proj.weight", torch_to_f16_np(normalize_std(qkv, TARGET_WEIGHT_STD, rescale))))
            entries.append(TargetEntry(f"{b}.attn_proj.weight", torch_to_f16_np(normalize_std(attn_out, residual_std, rescale))))
            entries.append(TargetEntry(f"{b}.up_gate.weight", torch_to_f16_np(normalize_std(up_gate, TARGET_WEIGHT_STD, rescale))))
            entries.append(TargetEntry(f"{b}.down_proj.weight", torch_to_f16_np(normalize_std(down, residual_std, rescale))))
        entries.append(TargetEntry(f"{b}.qkv_proj.bias", np_to_f16_np(np.zeros((3 * TARGET_D_MODEL,), dtype=np.float32))))
        entries.append(TargetEntry(f"{b}.attn_proj.bias", np_to_f16_np(np.zeros((TARGET_D_MODEL,), dtype=np.float32))))
        entries.append(TargetEntry(f"{b}.up_gate.bias", np_to_f16_np(np.zeros((2 * TARGET_MLP,), dtype=np.float32))))
        entries.append(TargetEntry(f"{b}.down_proj.bias", np_to_f16_np(np.zeros((TARGET_D_MODEL,), dtype=np.float32))))

    metadata = make_metadata(args, cfg, mapped_tokens)
    args._metadata = metadata  # keep simple for main()
    return entries


# Used only to avoid threading args through tiny projection helpers.
args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    if args.layers <= 0:
        raise SystemExit("--layers must be positive")
    model_dir = resolve_model_dir(args)
    print(f"model_dir: {model_dir}")
    entries = build_entries(args, model_dir)
    out = pathlib.Path(args.out)
    write_gdckpt(entries, out, args._metadata)
    total = sum(e.array.nbytes for e in entries)
    print(f"wrote {out} ({len(entries)} tensors, {total / 1e6:.2f} MB tensor payload)")


if __name__ == "__main__":
    main()
