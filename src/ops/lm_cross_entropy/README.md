# lm_cross_entropy

Fused language-model head plus cross entropy for tied-token embeddings:

```text
hidden [..., D], weight [V, D], targets [rows] -> scalar F32 loss
```

The training path computes logits with `hidden @ weight^T`, evaluates row-wise cross
entropy, and records one autograd node for the LM head and loss together.  For the
GPT-sized vocabulary used by `examples/gpt_lm`, the op streams vocab chunks through
F16 logits scratch and saves only compact row softmax stats (`row_max`, reciprocal
`row_sum`) for backward. Backward recomputes each logits chunk, forms `dlogits`, and
uses the production GEMM path for hidden/weight gradients.

`gd_lm_cross_entropy_softcapped(..., logits_softcap, ...)` optionally applies final
logits softcapping inside the fused loss without materializing full logits:

```text
soft_logit = logits_softcap * tanh(logit / logits_softcap)
```

`logits_softcap = 0` disables the transform and preserves the exact
`gd_lm_cross_entropy` behavior. When enabled, backward multiplies the cross-entropy
gradient by `1 - (soft_logit / logits_softcap)^2` before the gradient GEMMs.
