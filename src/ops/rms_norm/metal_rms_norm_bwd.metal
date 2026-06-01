#include "metal_common.metal"

kernel void gd_rms_norm_bwd(device const float *x               [[buffer(0)]],
                            device const float *weight          [[buffer(1)]],
                            device const float *go              [[buffer(2)]],
                            device float *dx                    [[buffer(3)]],
                            constant gd_metal_rmsnorm_params &p  [[buffer(4)]],
                            uint tgid  [[threadgroup_position_in_grid]],
                            uint tid    [[thread_index_in_threadgroup]],
                            uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows) {
        return;
    }
    threadgroup float pss[GD_RMS_TG];
    threadgroup float pa[GD_RMS_TG];
    int base = r * p.last;
    float lss = 0.0f;
    float la = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = x[base + c];
        lss += v * v;
        la += go[base + c] * weight[c] * v;
    }
    pss[tid] = lss;
    pa[tid] = la;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            pss[tid] += pss[tid + stride];
            pa[tid] += pa[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(pss[0] / (float)p.last + p.eps);
    float inv3 = inv * inv * inv;
    float A = pa[0];
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        dx[base + c] = inv * go[base + c] * weight[c] - x[base + c] * inv3 * A / (float)p.last;
    }
}
kernel void gd_rms_norm_bwd_add(device const float *x               [[buffer(0)]],
                                device const float *weight          [[buffer(1)]],
                                device const float *go              [[buffer(2)]],
                                device const float *ds_out          [[buffer(3)]],
                                device float *dx                    [[buffer(4)]],
                                device float *dsum                  [[buffer(5)]],
                                constant gd_metal_rmsnorm_params &p  [[buffer(6)]],
                                uint tgid  [[threadgroup_position_in_grid]],
                                uint tid    [[thread_index_in_threadgroup]],
                                uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows) {
        return;
    }
    threadgroup float pss[GD_RMS_TG];
    threadgroup float pa[GD_RMS_TG];
    int base = r * p.last;
    float lss = 0.0f;
    float la = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = x[base + c];
        lss += v * v;
        la += go[base + c] * weight[c] * v;
    }
    pss[tid] = lss;
    pa[tid] = la;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            pss[tid] += pss[tid + stride];
            pa[tid] += pa[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(pss[0] / (float)p.last + p.eps);
    float inv3 = inv * inv * inv;
    float A = pa[0];
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float d = inv * go[base + c] * weight[c] - x[base + c] * inv3 * A / (float)p.last;
        dx[base + c] = d;
        dsum[base + c] = d + ds_out[base + c];
    }
}
