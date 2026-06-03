# PyTorch op refs

PEP 723 scripts generate deterministic PyTorch references for gradients.c ops.
Run with `uv run` so deps install per script.

```sh
uv run tests/refs/torch_rms_norm_qkv_ref.py --out /tmp/rms_norm_qkv_ref.npz
uv run tests/refs/torch_sdpa_varlen_ref.py --out /tmp/sdpa_varlen_ref.npz
uv run tests/refs/torch_lm_cross_entropy_ref.py --out /tmp/lmce_ref.npz
uv run tests/refs/torch_powlu_ref.py --out /tmp/powlu_ref.npz
```

Run fp16 PyTorch ↔ gradients.c CPU forward/backward parity:

```sh
uv run tests/refs/compare_fp16_ops.py --device cpu
```

If a C/Metal harness dumps actual arrays with matching NPZ keys, compare:

```sh
uv run tests/refs/torch_rms_norm_qkv_ref.py --candidate /tmp/actual.npz
```
