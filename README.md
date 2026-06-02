# gradients.c

`gradients.c` is a small C tensor/autograd runtime focused on speed and ease
of extension: new ops, new architectures, new attention mechanisms, and new
backend kernels should be straightforward to add. It provides graph capture,
CPU reference execution, and Metal acceleration, with operator definitions,
shape/meta logic, autograd rules, and backend kernels kept in per-op capsules.

## Adding a new operator

Operators live as per-op capsules under `src/ops/<op>/`. Registries are generated
from filenames by `tools/gen_ops.c`; do not hand-edit `build/generated/*`.

### Fast path: scaffold stubs

Use the scaffold tool for new ops:

```sh
make tools
build/tools/gradients-new-op my_op
```

Defaults match the common case:

- public API: yes
- differentiable: yes
- custom backward op: yes
- backends: CPU reference + accelerated backends (Metal today; CUDA/Vulkan later)
- inputs/outputs: `1 -> 1`
- runs `make generated` after writing buildable stubs

Generated stubs contain explicit `TODO` comments and return `GD_ERR_UNSUPPORTED`
until implementation is filled; they never fake math.

Useful variants:

```sh
build/tools/gradients-new-op debug_print --private --no-diff --cpu-only
build/tools/gradients-new-op square --no-custom-bwd
build/tools/gradients-new-op layer_norm --inputs 3
build/tools/gradients-new-op my_op --dry-run
```

Common files created for `build/tools/gradients-new-op my_op`:

```text
src/ops/my_op/core_my_op_fwd.c
src/ops/my_op/cpu_my_op_fwd.c
src/ops/my_op/grad_my_op.c
src/ops/my_op/core_my_op_bwd.c
src/ops/my_op/cpu_my_op_bwd.c
src/ops/my_op/metal_my_op_fwd.m
src/ops/my_op/metal_my_op_fwd.metal
src/ops/my_op/metal_my_op_bwd.m
src/ops/my_op/metal_my_op_bwd.metal
```

It also patches public declarations/tests and Metal kernel mapping when needed.

### Fill implementation

1. **Core/meta** (`core_*_fwd.c`, optional `core_*_bwd.c`)
   - validate dtype/device/shape/layout/attrs
   - infer output descs
   - set `_gd_op_def` flags (`GD_OPF_PUBLIC`, `GD_OPF_DIFF`, internal bwd, etc.)

2. **CPU reference** (`cpu_*.c`)
   - put CPU run/kernel body in the per-op file
   - keep it simple, correct, deterministic
   - do not add new per-op bodies to `src/backends/cpu_ref/cpu_kernels.c`

3. **Autograd** (`grad_*.c`)
   - if gradient composes from existing ops, emit those ops directly
   - if gradient needs custom kernels, emit the generated `*_bwd` op

4. **Accelerated backends** (`metal_*.m`, `.metal` today)
   - implement support checks with clear unsupported reasons
   - implement host encoder and shader/MPS path
   - prefer optimized backend kernels over CPU-shaped ports

5. **Tests**
   - shape/error checks
   - CPU numerics
   - autograd/backward numerics
   - F16/BF16/etc. coverage if supported
   - accelerated backend parity or explicit fallback behavior

### Regenerate and test

The scaffold tool runs this by default; run it manually after changing op files:

```sh
make generated
```

Then build/test:

```sh
make test GD_ENABLE_METAL=0
make test
```

Check `build/generated/op_matrix.md` to verify core/grad/backend coverage.
