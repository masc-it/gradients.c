# Metal performance tips

Guidelines for high-performance forward and backward kernels in `gradients.c`.

For end-to-end op scaffolding and registration, see [Register a new op](register_op.md).

## Kernel ownership

- Keep op kernels in the op capsule:
  - `src/ops/<op>/metal_<op>.m`
  - `src/ops/<op>/metal_<op>_types.h`
  - `src/ops/<op>/metal_<op>.metal`
- Put shared algorithms under `src/ops/_shared/<domain>/` only when multiple ops need them.
- Keep ABI structs owner-local and use `src/backends/metal/metal_abi.h` scalar aliases.

## Dispatch and PSOs

- Specialize hot kernels by dtype/shape class when the branch is known on the host.
  - Prefer `op_f16_kernel` / `op_f32_kernel` PSOs over runtime dtype branches.
  - F16 inputs with FP32 accumulation are usually the right training default.
- Choose threadgroup geometry on the host from tensor shape.
  - Small reductions should not launch 256 threads per row.
  - Large reductions should use enough simdgroups to keep the row parallel.
- Dispatch all kernels for an op inside the active scope command buffer; avoid immediate submit/wait in hot paths.
- Validate byte ranges, dtype, rank, count, contiguity, and strides before encoding.

## Forward/backward design

- Optimize the training pair, not just the forward kernel.
- Save compact forward statistics when they remove expensive backward work.
  - Good: row max, reciprocal sum, normalization factors.
  - Bad: full softmax or large intermediates unless reuse clearly dominates memory cost.
- Backward should avoid recomputing reductions already computed in forward when saved stats are cheap.
- Direct public backward may keep a recompute path; autograd should use the fast saved-state path.

## Fusion and memory traffic

- Fuse simple producer/consumer stages when it removes global memory round-trips.
- Do not materialize intermediates such as softmax when the final formula can be written directly.
- Count global reads/writes before optimizing arithmetic. Most kernels are bandwidth-limited.
- Avoid extra final reductions or duplicate dispatches; they are easy to miss and expensive at scale.
- Prefer contiguous contracts for hot ops unless non-contiguous support is explicitly required.

## Reductions

- Use SIMD reductions first (`simd_sum`, `simd_max`), then reduce across simdgroups through threadgroup memory.
- Keep threadgroup memory small and fixed-size.
- Make reduction width shape-adaptive:
  - one simdgroup for small reductions,
  - more simdgroups for thousands of elements.
- Use numerically stable reductions for softmax-like ops:
  1. max pass,
  2. shifted exp/sum pass,
  3. final expression.

## Numerics

- Accumulate F16 inputs in FP32 for loss, normalization, and reductions.
- Store scalar losses and saved stats as F32.
- Be explicit about invalid-input behavior. If GPU-side validation is omitted for speed, document NaN/zero-gradient behavior.
- Check tolerance separately for F16 output storage and FP32 accumulation.

## ABI and safety

- Zero-initialize ABI structs on the host.
- Add `_Static_assert(sizeof(args) == expected)` for every host/Metal ABI struct.
- Pass buffer offsets in bytes and cast inside the kernel.
- Keep all counts checked for overflow before casting to Metal grid sizes.
- Do not rely on unreachable dtype branches for safety; host validation should reject unsupported dtypes.

## Performance validation

- Add both correctness and performance coverage for every optimized op.
- Correctness:
  - C tests for edge cases and dtype contract.
  - PyTorch harnesses for forward and backward parity.
- Performance:
  - Put op-local probes at `src/ops/<op>/perf_test.c` and run them with `make op-perf OP=<op>`.
  - Measure forward and forward+backward separately.
  - Use optimized builds (`O3`, `NDEBUG`) for numbers.
  - Report shape, dtype, warmup/iters, wall time, and effective bandwidth.
- Compare against the previous implementation before claiming an improvement.

## Quick checklist

- [ ] Dtype-specialized PSOs for hot kernels.
- [ ] Shape-adaptive threadgroup sizing.
- [ ] FP32 accumulation for F16 math-sensitive paths.
- [ ] No materialized softmax or unnecessary intermediates.
- [ ] Saved forward stats for expensive backward reductions.
- [ ] No duplicate dispatches/reductions.
- [ ] Host validation covers rank/shape/stride/count/range.
- [ ] C tests + PyTorch fwd/bwd harnesses + perf probe.
