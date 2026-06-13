# powlu

PoWLU gated activation ported from v1.

```text
gate(z, m) = z * sigmoid(z)                         if z <= 0
           = z ** (m / (sqrt(z) + 1)) * sigmoid(z)  otherwise
out = x1 * gate(x2, m)
```

`m` must satisfy `0 < m < 10`.

## v2 contract

- Public API: `gd_powlu(ctx, x1, x2, m, out)`.
- Direct backward: `gd_powlu_backward(ctx, x1, x2, grad_out, m, grad_x1, grad_x2)`.
- Current implementation is optimized for contiguous F16 tensors.
- Forward and backward use dedicated F16 Metal kernels with scalar `m` in the op attrs / kernel ABI.
- Backward computes both input gradients in a single dispatch when both are requested.

## Validation

```sh
make build/tests/test_powlu
GRADIENTS_METALLIB=build/gradients.metallib build/tests/test_powlu
GD_POWLU_FWD_PROFILE=smoke uv run src/ops/powlu/fwd.py
GD_POWLU_BWD_PROFILE=smoke uv run src/ops/powlu/bwd.py
make op-perf OP=powlu
```
