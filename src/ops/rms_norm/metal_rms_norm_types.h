#ifndef GD_OP_RMS_NORM_METAL_TYPES_H
#define GD_OP_RMS_NORM_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_RMS_NORM_MAX_SIMDGROUPS 8U
#define GD_METAL_RMS_NORM_WGRAD_ROW_BLOCK 64U
#define GD_METAL_RMS_NORM_WGRAD_ROW_BLOCK_LARGE 128U
#define GD_METAL_RMS_NORM_WGRAD_CHANNELS 128U

typedef struct gd_metal_rms_norm_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 weight_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 grad_out_offset;
    gd_metal_u64 inv_rms_offset;
    gd_metal_u64 partial_offset;
    gd_metal_u64 rows;
    gd_metal_u64 cols;
    gd_metal_u64 row_blocks;
    float eps;
    gd_metal_u32 simdgroups;
    gd_metal_u32 wgrad_simdgroups;
    gd_metal_u32 wgrad_row_block;
    gd_metal_u32 reserved0;
} gd_metal_rms_norm_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_rms_norm_args) == 96U,
               "gd_metal_rms_norm_args ABI mismatch");
#endif

#endif /* GD_OP_RMS_NORM_METAL_TYPES_H */
