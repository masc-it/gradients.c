# relu

ReLU op capsule.

Checklist:

- [x] Public API generated in `include/gradients/ops_generated.h`
- [x] Forward validation/allocation/recording in `core_relu.c`
- [x] Backend dispatch in `metal_relu.m`
- [x] Op-local Metal ABI/kernel implementation in `metal_relu_types.h` / `metal_relu.metal`
- [x] Backward rule in `autograd_relu.c`
- [x] Forward PyTorch harness in `fwd.py`
- [x] Backward PyTorch harness in `bwd.py`
- [x] C tests under `tests/`
