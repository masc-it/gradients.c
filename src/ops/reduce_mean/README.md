# reduce_mean

All-elements and single-axis mean reductions.

Initial scope:

- [x] Contiguous F16/F32 input
- [x] Rank-0 scalar output for all-elements reduction
- [x] FP32 accumulation in Metal kernels
- [x] F16 all-elements scalar mean returns F32 for loss-quality numerics
- [x] F16 all-elements multi-stage partials are staged through F32 scratch
- [x] Autograd rule: broadcast scalar gradient divided by input element count
- [x] Single-axis reductions with `keepdims=true/false`
- [x] Dtype-specialized Metal reduction and broadcast PSOs
- [x] Shape-adaptive axis reductions, including last-axis fast paths
- [x] Vectorized last-axis and scalar-gradient backward broadcasts
- [x] Op-local performance probe
- [x] PyTorch forward/backward harnesses
