# gradients.c

gradients.c is a small C tensor/autograd runtime for graph capture, with first class accelerated backends. Ops live in per-op capsules under
`src/ops/<op>/`; generated registries wire core/meta, autograd and backend
implementations together.

## register new op

Use the scaffold tool when possible and follow the full guide:

- [docs/rules/add_new_op.md](docs/rules/add_new_op.md)
- [docs/guides/metal_tips.md](docs/guides/metal_tips.md)
