# Register a new op

`gradients.c` uses op capsules under `src/ops/<op>/` plus a generated registry. Do not hand-edit `src/ops/op_kind.h` or `src/ops/op_registry.c`; they are generated from the op capsules.

## Quick start

```sh
make tools
build/tools/gradients-new-op my_op
# same-shape/broadcast binary elementwise scaffold:
build/tools/gradients-new-op --binary my_op
# custom backend entry points instead of generated backend= stubs:
build/tools/gradients-new-op --binary --custom my_loss
```

Useful scaffold flags:

- `--binary` / `--unary`: choose generated public API shape.
- `--custom` or `--no-backend`: omit generated `backend=` metadata so the op can define custom backend entry points manually.
- `--f16-only`: annotate/scaffold for dtype-specialized F16 kernels.
- `--f16-f32-accum`: annotate/scaffold for F16 inputs with FP32 accumulation.
- `--save-stats`: annotate/scaffold for compact forward stats consumed by backward.
- `--reduction`: annotate/scaffold for shape-adaptive reduction kernels.

The op name must be snake case: `[a-z][a-z0-9_]*` with no double or trailing underscores.

The scaffold tool creates the capsule directory and starter files, then regenerates the registry. Example output:

```text
[create] src/ops/my_op/
[create] src/ops/my_op/op_my_op.def
[create] src/ops/my_op/core_my_op.c
[create] src/ops/my_op/autograd_my_op.c
[create] src/ops/my_op/metal_my_op.m
[create] src/ops/my_op/metal_my_op_types.h
[create] src/ops/my_op/metal_my_op.metal
[create] src/ops/my_op/fwd.py
[create] src/ops/my_op/bwd.py
[create] src/ops/my_op/perf_test.c
[create] src/ops/my_op/README.md
[ops-registry] generated src/ops/op_kind.h
[ops-registry] generated src/ops/op_registry.c
[ops-registry] touched build/.ops-registry
[ops-registry] ops=3
[done] registered op 'my_op' as GD_OP_MY_OP
```

If a file already exists, the tool prints `[exists]` and leaves it untouched.

## What gets generated

The scaffold creates these op-capsule files under `src/ops/<op>/`:

```text
src/ops/<op>/op_<op>.def      # op metadata for generated stubs
src/ops/<op>/core_<op>.c       # forward/public op implementation
src/ops/<op>/autograd_<op>.c   # backward rule capsule
src/ops/<op>/metal_<op>.m      # accelerated backend host/dispatch capsule scaffold
src/ops/<op>/metal_<op>_types.h # op-local Metal kernel ABI structs
src/ops/<op>/metal_<op>.metal  # op-local Metal kernels compiled into the metallib
src/ops/<op>/fwd.py            # PEP 723 PyTorch forward comparison template
src/ops/<op>/bwd.py            # PEP 723 PyTorch backward comparison template
src/ops/<op>/perf_test.c       # optimized op-local performance probe scaffold
```

It writes:

```text
src/ops/op_kind.h                     # GD_OP_<NAME> enum values
src/ops/op_registry.c                 # autograd rule lookup table
include/gradients/ops_generated.h     # public API prototypes from op metadata
src/core/backend_generated.h          # backend API prototypes from op metadata
src/backends/null/backend_generated.c # null backend unsupported stubs
src/backends/metal/metal_ops_generated.inc # Metal PSO registration snippets
```

Existing enum IDs are preserved when possible; new ops are appended. `make build` also runs the registry generator before compiling.

The `op_<op>.def` metadata controls generated public/backend stubs. Supported shapes are:

```text
api=unary
backend=unary
```

and:

```text
api=binary
backend=binary
```

Unary ops export `gd_<op>(ctx, x, out)` and `gd_<op>_backward(ctx, x, grad_out, grad_x)`. Binary ops export `gd_<op>(ctx, x, y, out)` and `gd_<op>_backward(ctx, x, y, grad_out, grad_x, grad_y)`; the generated backend stub is the forward elementwise entry point.

For custom public signatures, extend `tools/gen_ops.c` with a new metadata shape before implementing the op. For standard unary/binary public APIs with non-elementwise backend work, use `--custom` / `--no-backend` and add custom backend declarations manually.

Metal kernels should stay in the op capsule. `make build` compiles `.metal` files found under `src/backends/metal/`, `src/ops/`, and `src/optim/` into the offline `gradients.metallib`; the generated Metal PSO glue only cares about exported kernel names. See [Metal kernel capsule layout](metal_capsules.md) for ownership rules and [Metal performance tips](metal_tips.md) for high-performance fwd/bwd kernel guidance.

## Implement the forward op

Edit `src/ops/my_op/core_my_op.c`.

Typical forward implementation shape:

```c
#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

/* Public declaration is generated in include/gradients/ops_generated.h. */
gd_status gd_my_op(gd_context *ctx,
                   const gd_tensor *x,
                   gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    const gd_tensor *inputs[1];
    gd_tensor *outputs[1];

    /* 1. validate inputs */
    /* 2. allocate y */
    /* 3. dispatch backend work */

    y.is_leaf = false;

    inputs[0] = x;
    outputs[0] = &y;
    st = gd_autograd_record(ctx,
                            GD_OP_MY_OP,
                            inputs,
                            1U,
                            outputs,
                            1U,
                            NULL,
                            0U,
                            NULL,
                            0U);
    if (st != GD_OK) {
        return st;
    }

    *out = y;
    return GD_OK;
}
```

Use an attrs struct when the backward rule needs scalar/config metadata:

```c
typedef struct gd_my_op_attrs {
    uint32_t axis;
} gd_my_op_attrs;

const gd_my_op_attrs attrs = {.axis = axis};
gd_autograd_record(ctx, GD_OP_MY_OP, inputs, n_inputs, outputs, n_outputs,
                   &attrs, sizeof(attrs), saved, n_saved);
```

Use `saved` tensors only for tensors the backward rule must read and that are not already inputs/outputs.

## Implement the backward rule

Edit `src/ops/my_op/autograd_my_op.c`.

The scaffold starts as unsupported:

```c
static gd_status gd_my_op_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    (void)bwd;
    (void)node;
    return GD_ERR_UNSUPPORTED;
}
```

Replace it with a rule that:

1. Reads input/output/saved tensor snapshots from the tape.
2. Gets the output gradient with `gd_autograd_get_grad`.
3. Computes gradient contributions using backend/core backward helpers.
4. Accumulates into each required input with `gd_autograd_accumulate`.

Example skeleton:

```c
static gd_status gd_my_op_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;

    if (x == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }

    /* compute dx */

    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_my_op = {
    .kind = GD_OP_MY_OP,
    .name = "my_op",
    .backward = gd_my_op_autograd_backward,
};
```

## Implement the performance probe

Edit `src/ops/my_op/perf_test.c`. Keep performance probes op-local and run them with:

```sh
make op-perf OP=my_op
```

The scaffold is compile-valid and intentionally prints TODO rows until the op is implemented. Replace the scaffold loop with real timed cases using this structure:

1. Define realistic shape/dtype cases, including tails and production-sized activations.
2. Create one reusable context/model per case outside the timed loop.
3. Warm up before measuring.
4. Measure separate entry points:
   - forward only,
   - direct public backward helper when available,
   - autograd forward+backward for the training-path baseline.
5. Include public API overhead, Metal command submission, and synchronization in wall time.
6. Report `avg_ms` plus one of:
   - `logical_GiB/s` for memory-bound elementwise/reduction ops,
   - `TFLOP/s` or `GFLOP/s` for compute-bound ops.
7. Print shape, dtype contract, warmup, and iteration count in the output.

Use environment variables for reproducibility and selective runs:

```sh
GD_MY_OP_PERF_PROFILE=all GD_MY_OP_PERF_WARMUP=10 GD_MY_OP_PERF_ITERS=100 make op-perf OP=my_op
```

The generic `op-perf` target builds the library with `O3`/`NDEBUG`, compiles `src/ops/<op>/perf_test.c`, and sets `GRADIENTS_METALLIB` for the probe. `perf_test.c` files are excluded from the library source list so they can contain standalone `main()` functions.

## Regenerate manually

Normally `make build` regenerates the registry. To run only the registry step and update the build stamp:

```sh
make tools
build/tools/gen_ops --stamp build/.ops-registry
```

## Validate

At minimum:

```sh
make test
```

For differentiable ops, also add:

- a C test under `tests/`
- completed Python/PyTorch comparison harnesses in `src/ops/my_op/fwd.py` and `src/ops/my_op/bwd.py`
- an optimized op-local performance probe at `src/ops/my_op/perf_test.c`

Run Python harnesses with `uv run ...`. Run the op-local performance probe with:

```sh
make op-perf OP=my_op
```

## Checklist

- [ ] `build/tools/gradients-new-op my_op`
- [ ] Confirm generated public declarations in `include/gradients/ops_generated.h`
- [ ] Forward implementation in `src/ops/my_op/core_my_op.c`
- [ ] Backend dispatch in `src/ops/my_op/metal_my_op.m`
- [ ] Op-local Metal types/kernels in `src/ops/my_op/metal_my_op_types.h` and `src/ops/my_op/metal_my_op.metal`
- [ ] Backward rule in `src/ops/my_op/autograd_my_op.c`
- [ ] Forward/backward PyTorch harnesses in `src/ops/my_op/fwd.py` / `src/ops/my_op/bwd.py`
- [ ] C tests and PyTorch comparison
- [ ] Op-local perf probe in `src/ops/my_op/perf_test.c` (`make op-perf OP=my_op`)
- [ ] `make test`
