# relu

Generated op capsule scaffold.

Checklist:

- [ ] Public API declaration in `include/gradients/ops.h`
- [ ] Forward validation/allocation/recording in `core_relu.c`
- [ ] Backend dispatch in `metal_relu.m` + kernels if needed
- [ ] Backward rule in `autograd_relu.c`
- [ ] Forward PyTorch harness in `fwd.py`
- [ ] Backward PyTorch harness in `bwd.py`
- [ ] C tests under `tests/`
