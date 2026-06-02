# Add new operator

Use this guide when adding a new gradients.c op.

Operators are owned by per-op capsules under:

```text
src/ops/<op>/
```

Registry files are generated from capsule filenames by `tools/gen_ops.c`. Never edit
`build/generated/*` by hand.

## Fast path: scaffold op

Build tools, then scaffold:

```sh
make tools
build/tools/gradients-new-op my_op
```

Default scaffold matches common new-op work:

- public API: yes
- differentiable: yes
- custom backward op: yes
- inputs: 1
- outputs: 1
- backends: CPU reference + accelerated backends (Metal today; CUDA/Vulkan later)
- runs `make generated` after writing files

Generated code is compile-valid but not implemented. Stubs include `TODO` comments
and return `GD_ERR_UNSUPPORTED` until real math/meta/backend code is filled.
They must never fake numeric results.

## Tool options

```text
gradients-new-op OP_NAME [options]

options:
  --private         skip public API and public-symbol test patches
  --no-diff         forward-only op; skip autograd and backward stubs
  --no-custom-bwd   add grad rule stub only; skip explicit *_bwd op stubs
  --inputs N        forward input count (default: 1)
  --outputs N       forward output count (default: 1)
  --cpu-only        skip accelerated backend stubs
  --no-generated    do not run `make generated` after scaffolding
  --force           overwrite existing scaffold files
  --dry-run         print plan without writing files or running make
  --help            show usage
```

Examples:

```sh
# Common public differentiable op, with custom backward + Metal stubs.
build/tools/gradients-new-op softplus

# Differentiable op whose backward composes existing ops, no custom *_bwd op.
build/tools/gradients-new-op square --no-custom-bwd

# Private CPU-only debug op.
build/tools/gradients-new-op debug_print --private --no-diff --cpu-only

# Multi-input op.
build/tools/gradients-new-op layer_norm --inputs 3

# Preview only.
build/tools/gradients-new-op softplus --dry-run
```

## Files and roles

Common scaffold for `my_op`:

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

### `core_<op>_fwd.c`

Core forward definition.

Owns:

- `_gd_opdef_<op>`
- public `gd_<op>(...)` wrapper when op is public
- forward meta function

Implement:

- input/output count checks
- dtype/device/layout/shape validation
- attrs validation
- output desc inference
- flags: `GD_OPF_PUBLIC`, `GD_OPF_DIFF`, etc.

No numeric kernel math belongs here.

### `cpu_<op>_fwd.c`

CPU reference forward executor.

Owns:

- `_gd_cpu_op_<op>`
- CPU support/run functions

Implement CPU forward math here, or in a private same-capsule helper such as
`kernel_<op>.c` if fwd/bwd share logic.

Do not add new per-op kernel bodies to:

```text
src/backends/cpu_ref/cpu_kernels.c
```

CPU ref is being aligned with accelerated backend layout: op-specific code lives
inside `src/ops/<op>/`.

### `grad_<op>.c`

Autograd rule for forward op.

Owns:

- `_gd_bwd_rule_<op>`

Implement one of:

- compose gradient from existing ops; or
- emit custom `_GD_OP_<OP>_BWD` and accumulate returned grads.

### `core_<op>_bwd.c`

Core definition for custom backward op.

Owns:

- `_gd_opdef_<op>_bwd`

Backward op is internal, not public API. Implement bwd meta validation and grad
output desc inference.

### `cpu_<op>_bwd.c`

CPU reference backward executor.

Owns:

- `_gd_cpu_op_<op>_bwd`

Implement CPU backward math here, or share same-capsule helpers.

### `metal_<op>_fwd.m` / `metal_<op>_bwd.m`

Metal host-side backend capsules.

Own:

- `_gd_metal_op_<op>`
- `_gd_metal_op_<op>_bwd`

Implement:

- support checks with clear unsupported reasons
- plan setup if needed
- command encoder setup
- MPS or shader dispatch

### `metal_<op>_fwd.metal` / `metal_<op>_bwd.metal`

Metal shader code compiled into `build/gradients.metallib`.

Use optimized GPU kernels. Do not copy CPU-shaped loops unless acceptable for a
small/simple op.

## Generated registry

`make generated` scans filenames and writes:

```text
build/generated/op_kind.h
build/generated/op_registry.inc
build/generated/bwd_registry.inc
build/generated/cpu_registry.inc
build/generated/metal_registry.inc
build/generated/metal_shaders.mk
build/generated/op_matrix.md
```

The scaffold tool runs `make generated` by default. Run it manually after adding,
renaming, or deleting op capsule files:

```sh
make generated
```

Check coverage:

```sh
less build/generated/op_matrix.md
```

## Backend policy

- CPU reference is required for correctness and fallback.
- Accelerated backend stubs are default; use `--cpu-only` only for ops that must
  not run on accelerated backends.
- If accelerated implementation is incomplete, return `GD_ERR_UNSUPPORTED` with
  an exact message. Do not silently fall back inside backend code; let runtime
  fallback policy handle fallback.
- Future accelerated backends (CUDA/Vulkan) should follow same per-op capsule
  pattern.

## Test checklist

Add focused tests under `tests/` or extend existing tests:

- public symbol test when public
- op registry coverage
- shape/dtype/device error cases
- CPU forward numerics
- backward/autograd numerics
- F16/BF16/etc. cases when supported
- accelerated backend parity or explicit unsupported/fallback behavior

Run:

```sh
make test GD_ENABLE_METAL=0
make test
```

For Metal-specific checks, ensure metallib is available:

```sh
GRADIENTS_METALLIB=build/gradients.metallib build/tests/<test>
```
