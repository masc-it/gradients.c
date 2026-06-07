# lm_cross_entropy

Fused language-model head plus cross entropy for tied-token embeddings:

```text
hidden [..., D], weight [V, D], targets [rows] -> scalar F32 loss
```

The training path computes logits with `hidden @ weight^T`, evaluates row-wise cross
entropy, and records one autograd node for the LM head and loss together.  For the
GPT-sized vocabulary used by `examples/gpt_lm`, the op saves the F16 logits plus
compact row softmax stats so backward can reuse the production GEMM path without
recomputing the LM head. This keeps math identical to the materialized
`linear_transposed_weight + cross_entropy` sequence while trimming tape overhead and
releasing temporary row-loss / dlogits scratch as soon as the fused op no longer
needs it.
