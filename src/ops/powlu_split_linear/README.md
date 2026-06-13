# powlu_split_linear

Generated op capsule scaffold. Before implementing Metal hot paths, read
[Metal performance tips](../../../docs/guides/metal_tips.md).

Scaffold notes:

- See docs/guides/metal_tips.md before implementing Metal hot paths.
- Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.

Checklist:

- [ ] Public API generated in `include/gradients/ops_generated.h`
- [ ] Forward validation/allocation/recording in `core_powlu_split_linear.c`
- [ ] Backend dispatch in `metal_powlu_split_linear.m`
- [ ] Op-local Metal ABI/kernel implementation in `metal_powlu_split_linear_types.h` / `metal_powlu_split_linear.metal`
- [ ] Backward rule in `autograd_powlu_split_linear.c`
- [ ] Forward PyTorch harness in `fwd.py`
- [ ] Backward PyTorch harness in `bwd.py`
- [ ] C tests under `tests/`
- [ ] Op-local perf probe in `perf_test.c` (`make op-perf OP=powlu_split_linear`)
