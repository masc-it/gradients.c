#ifndef GD_OPTIM_ADAMW_METAL_TYPES_H
#define GD_OPTIM_ADAMW_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

typedef struct gd_metal_adamw_args {
    gd_metal_u64 param_offset;
    gd_metal_u64 master_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 m_offset;
    gd_metal_u64 v_offset;
    gd_metal_u64 count;
    gd_metal_u32 param_dtype;
    gd_metal_u32 grad_dtype;
    gd_metal_u32 has_master;
    gd_metal_u32 pad0;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
    float pad1;
} gd_metal_adamw_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_adamw_args) == 96U, "gd_metal_adamw_args ABI mismatch");
_Static_assert(offsetof(gd_metal_adamw_args, lr) == 64U, "gd_metal_adamw_args lr offset mismatch");
#endif

#endif /* GD_OPTIM_ADAMW_METAL_TYPES_H */
