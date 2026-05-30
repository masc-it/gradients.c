#include <metal_stdlib>
using namespace metal;

/* Placeholder shader to validate the .metal -> .air -> .metallib build path. */
kernel void gd_probe_identity(device float *x [[buffer(0)]],
                              uint i [[thread_position_in_grid]])
{
    x[i] = x[i];
}
