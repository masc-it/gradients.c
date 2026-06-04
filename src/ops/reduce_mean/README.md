# reduce_mean

All-elements and single-axis mean reductions.

Initial scope:

- [x] Contiguous F16/F32 input
- [x] Rank-0 scalar output for all-elements reduction
- [x] FP32 accumulation in Metal kernel, result stored in input dtype
- [x] Multi-stage contiguous reduction for large tensors
- [x] Autograd rule: broadcast scalar gradient divided by input element count
- [x] Single-axis reductions with `keepdims=true/false`
- [x] Optimized Metal axis reduction and reduced-axis backward broadcast
- [ ] PyTorch harness
