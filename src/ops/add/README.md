# add

Binary elementwise add op capsule.

Initial scope:

- [x] Same-shape contiguous F16 optimized fast path
- [x] NumPy-style broadcasting for contiguous F16 tensors
- [x] Autograd rule: `dx = sum_to_shape(grad_out, x)`, `dy = sum_to_shape(grad_out, y)`
- [x] Op-local vectorized Metal kernel capsule
- [x] Op-local performance probe
- [ ] Scalar convenience APIs
- [ ] PyTorch harness
