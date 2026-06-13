# gradients.c

> The library is in active development and not yet released to public.

`gradients.c` is a C tensor/autograd runtime focused on speed and ease
of extension: new ops, new architectures, new attention mechanisms, and new
backend kernels should be straightforward to add. 

Supported backends as of now: Metal

## Adding a new operator

Operators live as per-op capsules under `src/ops/<op>/`. Registries are generated
from filenames by `tools/gen_ops.c`; do not hand-edit `build/generated/*`.

More details in docs/guides/register_op.md