#ifndef GRADIENTS_METAL_KERNEL_TYPES_H
#define GRADIENTS_METAL_KERNEL_TYPES_H

/* Shared layout contract between the Objective-C host (metal_backend.m) and the
 * Metal shading language kernels (kernels.metal). Keep this POD and restricted
 * to types valid in both C and MSL (plain int/float). Sizes are element counts;
 * strides are not needed because v1 produced values are contiguous and inputs
 * broadcast by shape, matching the CPU reference kernels. */

#define GD_METAL_MAX_DIMS 8

/* Elementwise binary ops (add/mul) with NumPy-style right-aligned broadcasting.
 * `out_sizes` describes the contiguous output; `a_sizes`/`b_sizes` describe each
 * input's own shape so the shader can reproduce broadcast_offset() from the CPU
 * kernel exactly. */
typedef struct gd_metal_ew_params {
    int ndim;                          /* output rank */
    int numel;                         /* output element count */
    int a_ndim;
    int b_ndim;
    int out_sizes[GD_METAL_MAX_DIMS];
    int a_sizes[GD_METAL_MAX_DIMS];
    int b_sizes[GD_METAL_MAX_DIMS];
} gd_metal_ew_params;

#endif /* GRADIENTS_METAL_KERNEL_TYPES_H */
