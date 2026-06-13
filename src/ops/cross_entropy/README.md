# cross_entropy

Mean sparse softmax cross entropy for logits and integer class labels.

Contract:

- `gd_cross_entropy(ctx, logits, targets, out)` computes scalar mean loss.
- `logits`: contiguous `[N, C]`, dtype F16. Accumulation and scalar loss are F32.
- `targets`: contiguous `[N]`, dtype I32 class indices in `[0, C)`.
- `out`: rank-0 F32 scalar.
- Backward produces `grad_logits` with the same dtype/shape as `logits`.
- Targets are non-differentiable.

Implementation notes:

- Forward uses an F16-specialized op-local Metal row-loss kernel that computes
  stable `logsumexp(logits[n, :]) - logits[n, target[n]]` with FP32
  accumulation.
- In training, forward also saves per-row `row_max` and `row_inv_sum` for the
  autograd fast path.
- A shared reduction kernel averages row losses to a scalar F32 loss.
- Autograd backward consumes saved row stats and writes
  `(softmax(logits) - one_hot(target)) * grad_loss / N` without materializing
  softmax or transposes. The public direct backward still has a recompute path.
- Row-reduction kernels pick 1/2/4/8 simdgroups per row based on the class
  count, avoiding the old fixed 256-thread footprint for small vocabularies.

Checklist:

- [x] Public API generated in `include/gradients/ops_generated.h`
- [x] Forward validation/allocation/recording in `core_cross_entropy.c`
- [x] Op-local Metal ABI/kernel implementation
- [x] Backward rule in `autograd_cross_entropy.c`
- [x] C tests under `tests/`
- [x] Forward/backward PyTorch harnesses
