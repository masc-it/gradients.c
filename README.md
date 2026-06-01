# gradients.c

`gradients.c` is a small C tensor/autograd runtime focused on speed and ease
of extension: new ops, new architectures, new attention mechanisms, and new
backend kernels should be straightforward to add. It provides graph capture,
CPU reference execution, and Metal acceleration, with operator definitions,
shape/meta logic, autograd rules, and backend kernels kept in per-op capsules.

## Adding a new operator

Operators live as small capsules under `src/ops/<op>/`. Registries are generated
from filenames by `tools/gen_ops.c`, so names must follow existing grammar.

### 1. Add public API

Add declaration to `include/gradients/ops.h` if op is public:

```c
gd_status gd_my_op(gd_context *ctx, gd_tensor *x, gd_tensor **out);
```

### 2. Add core op capsule

Create `src/ops/my_op/core_my_op_fwd.c`.

Core capsule owns:

- `_gd_op_def _gd_opdef_my_op`
- meta function: validate dtype/device/shape/layout and infer output desc
- public wrapper that calls `_gd_emit_checked()`

Pattern:

```c
#include "../op_impl.h"
#include "../meta_common.h"
#include "gradients/ops.h"

#include "../../core/internal.h"

static gd_status my_op_meta(const gd_tensor_desc *const *inputs,
                            int n_inputs,
                            _gd_op_attrs *attrs,
                            gd_tensor_desc *outputs,
                            int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    (void)attrs;
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_unary_float(inputs[0], &outputs[0]);
}

const _gd_op_def _gd_opdef_my_op = {
    .kind = _GD_OP_MY_OP,
    .name = "my_op",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = my_op_meta,
};

gd_status gd_my_op(gd_context *ctx, gd_tensor *x, gd_tensor **out)
{
    gd_tensor *inputs[1] = {x};
    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_my_op argument is NULL");
    }
    *out = NULL;
    return _gd_emit_checked(ctx, _GD_OP_MY_OP, inputs, 1, NULL, out, 1);
}
```

`_GD_OP_MY_OP` appears after `make generated`; do not hand-edit generated files.

### 3. Add CPU reference support

Create `src/ops/my_op/cpu_my_op_fwd.c` with `_gd_cpu_op _gd_cpu_op_my_op`.
CPU ref should be simple, correct, and deterministic. Reuse helpers from
`src/backends/cpu_ref/cpu_backend.h` when possible.

### 4. Add autograd rule if differentiable

Create `src/ops/my_op/grad_my_op.c` with `_gd_bwd_rule _gd_bwd_rule_my_op`.
If op is not differentiable, omit `GD_OPF_DIFF` and no grad file needed.

### 5. Add non-CPU backend support

For Metal or future non-CPU backends (CUDA, Vulkan), prefer optimized implementations instead
of direct CPU-shaped ports. If no optimized backend implementation exists, leave
op unsupported and let CPU fallback handle it.

Metal support usually needs:

- `src/ops/my_op/metal_my_op_fwd.m` host capsule with `_gd_metal_op_my_op`
- `src/ops/my_op/metal_my_op_fwd.metal` shader kernel when using custom shader
- support checks that reject unsupported dtype/layout/shape with clear reason

Use existing ops such as `add`, `matmul`, or `sdpa` as templates.

### 6. Regenerate and test

```sh
make generated
make test GD_ENABLE_METAL=0
make test
```

Also add focused tests under `tests/` for shape errors, CPU numerics, autograd
(if any), and backend parity/fallback behavior.
