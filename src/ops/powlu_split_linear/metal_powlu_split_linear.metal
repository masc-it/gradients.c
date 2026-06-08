#include <metal_stdlib>
#include "metal_powlu_split_linear_types.h"

using namespace metal;

/* Scaffold for the 'powlu_split_linear' op-local Metal kernels.

   The generated Metal PSO glue expects unary ops to export:
     gd_powlu_split_linear_kernel
     gd_powlu_split_linear_backward_kernel

   See docs/guides/metal_tips.md before implementing Metal hot paths.
   Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
*/
kernel void gd_powlu_split_linear_kernel(device const uchar *x [[buffer(0)]],
                         device uchar *y [[buffer(1)]],
                         constant gd_metal_powlu_split_linear_args &args [[buffer(2)]],
                         uint gid [[thread_position_in_grid]])
{
    (void)x;
    (void)y;
    (void)args;
    (void)gid;
}

kernel void gd_powlu_split_linear_backward_kernel(device const uchar *x [[buffer(0)]],
                                  device const uchar *grad_out [[buffer(1)]],
                                  device uchar *grad_x [[buffer(2)]],
                                  constant gd_metal_powlu_split_linear_args &args [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]])
{
    (void)x;
    (void)grad_out;
    (void)grad_x;
    (void)args;
    (void)gid;
}
