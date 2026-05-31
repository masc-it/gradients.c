# gradients.c — Fused LM Head + Cross Entropy

Status: implemented opt-in v1 (2026-05-31)

Goal: avoid materializing full LM logits and `dlogits` for tied-LM-head training:

```text
hidden[...,D] @ weight[V,D]^T -> logits[...,V] -> mean cross_entropy(targets)
```

The fused op computes the same scalar loss and gradients for `hidden` and
`weight`, but only keeps a `[N, C]` vocab chunk in scratch (`N=B*T`,
`C=GD_METAL_LMCE_CHUNK`, default 1024). CPU_REF remains the oracle.

---

## 1. Why this exists

The unfused training graph materializes both:

```text
logits  [B,T,V]
dlogits [B,T,V]
```

Memory cost:

```text
2 * B * T * V * sizeof(float)
```

Examples at `V=8000`:

```text
B=8,T=512:   logits+dlogits ~= 262 MB
B=8,T=1024:  logits+dlogits ~= 524 MB
```

With vocab 50k, the same region becomes multi-GB. The fused path bounds scratch
to:

```text
N * C * sizeof(float) + O(N)
```

At `C=1024`:

```text
B=8,T=512:   chunk scratch ~= 16 MB
B=8,T=1024:  chunk scratch ~= 32 MB
```

---

## 2. Public/API surface

Added:

```c
gd_status gd_lm_cross_entropy(gd_context *ctx,
                              gd_tensor *hidden,   /* [..., D] */
                              gd_tensor *weight,   /* [V, D] */
                              gd_tensor *targets,  /* hidden shape without D */
                              gd_tensor **loss);
```

Added GPT convenience:

```c
gd_status gd_gpt_forward_loss(gd_context *ctx, gd_gpt *gpt,
                              gd_tensor *tokens, gd_tensor *positions,
                              gd_tensor *targets, gd_tensor **loss_out);
```

For tied embeddings, `gd_gpt_forward_loss` uses `gd_lm_cross_entropy`; for untied
heads it falls back to `gd_linear + gd_cross_entropy`.

Benchmark harness toggle:

```sh
GD_METAL_MPS=1 GD_BENCH_FUSED_LMCE=1 make gpt-bench
```

Metal LMCE v1 uses MPS GEMMs internally and therefore requires the existing
`GD_METAL_MPS=1` opt-in gate. CPU_REF works without that env var.

The old logits-returning `gd_gpt_forward` remains unchanged for generation,
evaluation, and callers that need logits.

---

## 3. CPU_REF implementation

Forward:

```text
for row n:
  max = max_v dot(hidden[n], weight[v])
  sum = Σ_v exp(dot - max)
  loss += -(target_logit - max - log(sum))
loss /= N
```

Backward:

```text
dhidden = 0
dweight = 0
for row n:
  recompute max/sum
  for class v:
    dlogit = (softmax_v - onehot_v) * go / N
    dhidden[n] += dlogit * weight[v]
    dweight[v] += dlogit * hidden[n]
```

No logits/dlogits tensor is allocated.

---

## 4. Metal implementation

Metal uses MPS for GEMMs and small custom kernels for softmax stats / dlogits.

Forward per vocab chunk:

```text
MPS:    logits_chunk = hidden_flat @ weight_chunk^T
kernel: update row-wise online max/sum and target_logit
```

After all chunks:

```text
kernel: losses[row] = -(target_logit - m - log(sum))
kernel: reduce mean loss
```

Backward:

```text
# recompute row stats
for chunk:
  MPS:    logits_chunk = hidden_flat @ weight_chunk^T
  kernel: update m/sum

# emit gradients without full dlogits
for chunk:
  MPS:    logits_chunk = hidden_flat @ weight_chunk^T
  kernel: overwrite logits_chunk with dlogits_chunk
  MPS:    dhidden += dlogits_chunk @ weight_chunk
  MPS:    dweight_chunk = dlogits_chunk^T @ hidden_flat
```

`dhidden` uses MPS `beta=0` for the first chunk and `beta=1` for later chunks.
`dweight_chunk` writes disjoint weight rows, so no accumulation between chunks is
needed.

Scratch layout:

```text
logits_chunk [N, C]
row_max      [N]
row_sum      [N]
target_logit [N]
row_losses   [N]
```

This uses the existing shared Metal scratch arena, so peak scratch is the max
of all op scratch sizes, not a sum over nodes.

---

## 5. Validation

Validated after implementation:

```text
make check
GD_METAL_MPS=1 make check
make docs-check
ASan test_metal_gpt with GD_METAL_MPS=1
```

New coverage:

- CPU finite-difference gradcheck for `gd_lm_cross_entropy` (`hidden`, `weight`).
- Metal forward parity for fused loss.
- Metal backward parity for `dhidden` and `dweight`.
- GPT bench can run fused path with `GD_BENCH_FUSED_LMCE=1`.

---

## 6. Performance notes

Bench environment: release gpt-bench, `GD_METAL_MPS=1`, `GD_BENCH_FUSED_LMCE=1`, model default
`d_model=320 layers=6 heads=5 d_ff=1280 vocab=8000`.

Baseline before fused LMCE (MPS + scratch arena + SDPA tuning):

```text
T256 B4:  ~192 ms, ~5334 tok/s
T512 B8: ~1098 ms, ~3729 tok/s
T1024 B4 smoke: ~1965 ms, ~2084 tok/s
```

Fused LMCE v1 (`C=1024`):

```text
T256 B4:  229.0 ms best, 4472 tok/s
T512 B8: 1145.0 ms best, 3577 tok/s
T1024 B4 smoke: 2046.7 ms, 2001 tok/s
```

Chunk-size probe at `T512 B8`:

```text
C=512 : 1156.4 ms best
C=1024: 1149.5 ms best
C=2048: 1151.4 ms best
C=4096: 1149.6 ms best
C=8000: 1153.3 ms best
```

Conclusion: v1 is a memory-headroom feature, not a speed win at vocab=8000.
Cost comes from repeated chunk GEMMs and kernels. Keep opt-in in benchmark until
larger-vocab/headroom workloads prove net-positive.

---

## 7. Follow-ups

- Consider direct fused row-softmax+dweight/dhidden kernels for small vocab where
  MPS chunk overhead dominates.
- Consider chunk size env/runtime knob if larger vocab models prefer different
  memory/speed tradeoff.
- Consider in-place optimizer/path-level planner that chooses fused LMCE only when
  logits are not otherwise consumed and memory pressure is high.
