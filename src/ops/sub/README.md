# sub

Binary elementwise subtract op capsule.

Initial scope:

- [x] Same-shape contiguous F16/F32 fast path
- [x] NumPy-style broadcasting for contiguous F16/F32 tensors
- [x] Autograd rule: `dx = sum_to_shape(grad_out, x)`, `dy = -sum_to_shape(grad_out, y)`
- [x] Op-local Metal kernel capsule
- [ ] Scalar convenience APIs
- [ ] PyTorch harness
