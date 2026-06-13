# reduce_sum

All-elements and single-axis sum reductions.

Initial scope:

- [x] Contiguous F16/F32 input
- [x] Rank-0 scalar output for all-elements reduction
- [x] FP32 accumulation in Metal kernels; F16 all-reduce partials are staged in F32 scratch
- [x] Multi-stage contiguous reduction for large tensors
- [x] Autograd rule: broadcast scalar gradient back to input shape
- [x] Single-axis reductions with `keepdims=true/false`
- [x] Dtype-specialized Metal PSOs for all reduce/broadcast kernels
- [x] Shape-adaptive SIMD-group reductions, including last-axis fast paths
- [x] Specialized scalar-gradient broadcast for all-elements backward
- [x] Op-local performance probe
- [x] PyTorch forward/backward harnesses
