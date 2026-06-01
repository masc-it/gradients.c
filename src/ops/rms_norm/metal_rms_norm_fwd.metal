#include "metal_common.metal"

kernel void gd_rms_norm(device const uchar *x              [[buffer(0)]],
                        device const uchar *weight         [[buffer(1)]],
                        device uchar *out                  [[buffer(2)]],
                        constant gd_metal_rmsnorm_params &p [[buffer(3)]],
                        uint tgid  [[threadgroup_position_in_grid]],
                        uint tid    [[thread_index_in_threadgroup]],
                        uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows) {
        return;
    }
    threadgroup float part[GD_RMS_TG];
    int base = r * p.last;
    float local = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = gd_load_float(x, p.dtype, (uint)(base + c));
        local += v * v;
    }
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            part[tid] += part[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(part[0] / (float)p.last + p.eps);
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float xv = gd_load_float(x, p.dtype, (uint)(base + c));
        float wv = gd_load_float(weight, p.dtype, (uint)c);
        gd_store_float(out, p.dtype, (uint)(base + c), xv * inv * wv);
    }
}
kernel void gd_add_rms_norm(device const float *a               [[buffer(0)]],
                            device const float *b               [[buffer(1)]],
                            device const float *weight          [[buffer(2)]],
                            device float *sum_out               [[buffer(3)]],
                            device float *norm_out              [[buffer(4)]],
                            constant gd_metal_rmsnorm_params &p  [[buffer(5)]],
                            uint tgid  [[threadgroup_position_in_grid]],
                            uint tid    [[thread_index_in_threadgroup]],
                            uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows || p.last > GD_METAL_FUSED_RMS_MAX) {
        return;
    }
    threadgroup float part[GD_RMS_TG];
    threadgroup float sum_sh[GD_METAL_FUSED_RMS_MAX];
    int base = r * p.last;
    float local = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float s = a[base + c] + b[base + c];
        sum_sh[c] = s;
        sum_out[base + c] = s;
        local += s * s;
    }
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            part[tid] += part[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(part[0] / (float)p.last + p.eps);
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        norm_out[base + c] = sum_sh[c] * inv * weight[c];
    }
}
