# Metal kernel capsule layout

Metal kernels are owned by the subsystem that uses them. Avoid catch-all shader or ABI files.

## Ownership

- Op kernels live in `src/ops/<op>/`.
  - Host dispatch: `metal_<op>.m`
  - Kernel ABI: `metal_<op>_types.h`
  - Shaders: `metal_<op>.metal`
- Shared op algorithms live under `src/ops/_shared/<domain>/`.
  - Example: GEMM helpers in `src/ops/_shared/gemm/`.
- Backend primitives live under `src/backends/metal/primitives/<primitive>/`.
  - Example: fill/accumulate in `primitives/memory/`.
- Optimizer kernels live under `src/optim/<optimizer>/`.
  - Example: AdamW Metal ABI/kernel files in `src/optim/adamw/`.

## Shared ABI and helpers

Use `src/backends/metal/metal_abi.h` for host/Metal scalar aliases used in ABI structs.
Use `src/backends/metal/metal_common.h` only for small Metal-side helpers shared across backend primitives/optimizers.

Do not recreate a central `kernels.metal` or `metal_kernel_types.h`; add small owner-local files instead.

## Build

`make build` compiles every `.metal` file under:

```text
src/backends/metal/
src/ops/
src/optim/
```

All compiled AIR files are linked into the offline `build/gradients.metallib`.
