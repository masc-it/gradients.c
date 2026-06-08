#include <gradients/ops.h>

#include "../op_common.h"

/* Scaffold for the 'powlu_split_linear' op.

   Next steps:
   1. Confirm generated public declaration in include/gradients/ops_generated.h.
   2. Replace this anchor with gd_powlu_split_linear(...) validation/allocation/backend dispatch.
   3. Record the op with gd_autograd_record(ctx, GD_OP_POWLU_SPLIT_LINEAR, ...).
   4. Add backend implementations/tests/probes as needed.

   See docs/guides/metal_tips.md before implementing Metal hot paths.
   Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
*/
typedef int gd_powlu_split_linear_core_scaffold_anchor;
