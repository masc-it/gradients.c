# gradients.c

gradients.c is a small C tensor/autograd runtime for graph capture, CPU reference
execution, and accelerated backends. Ops live in per-op capsules under
`src/ops/<op>/`; generated registries wire core/meta, autograd, CPU, and backend
implementations together.

## register new op

Use the scaffold tool and follow the full guide:

- [docs/rules/add_new_op.md](docs/rules/add_new_op.md)

Quick start:

```sh
make tools
build/tools/gradients-new-op my_op
```

Default scaffold creates public differentiable fwd+bwd stubs for CPU ref and
accelerated backends. Stubs are compile-valid, marked TODO, and return
`GD_ERR_UNSUPPORTED` until implemented.
