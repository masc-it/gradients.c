#include "../../backends/metal/metal_backend_internal.h"
#include "metal_powlu_split_linear_types.h"

/* Custom Metal backend capsule for 'powlu_split_linear'.

   backend= is omitted in op_powlu_split_linear.def. Add custom backend declarations
   to src/core/backend.h and PSO creation/release in
   src/backends/metal/backend_metal.m when implementing this op.

   See docs/guides/metal_tips.md before implementing Metal hot paths.
   Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
*/
typedef int gd_powlu_split_linear_metal_scaffold_anchor;
