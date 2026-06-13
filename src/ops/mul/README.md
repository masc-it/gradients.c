# mul

Binary elementwise multiply op capsule.

Initial scope:

- [x] Same-shape contiguous F16 fast path
- [x] NumPy-style broadcasting for contiguous F16 tensors
- [x] Autograd rule: `dx = sum_to_shape(grad_out * y, x)`, `dy = sum_to_shape(grad_out * x, y)`
- [x] Op-local Metal kernel capsule
- [x] Vectorized F16 Metal forward kernels
- [x] Fused F16 direct backward kernel
- [x] Fused F16 suffix broadcast-gradient reduction
- [ ] Scalar convenience APIs
- [ ] PyTorch harness
