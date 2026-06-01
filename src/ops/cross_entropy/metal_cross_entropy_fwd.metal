#include "metal_common.metal"

kernel void gd_cross_entropy(device const float *logits      [[buffer(0)]],
                             device const int *targets       [[buffer(1)]],
                             device float *losses            [[buffer(2)]],
                             constant gd_metal_ce_params &p   [[buffer(3)]],
                             uint gid  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float red[GD_CE_TG];
    int total = p.outer * p.inner;
    int pos = (int)gid;
    if (pos >= total) {
        return;
    }
    int o = pos / p.inner;
    int in = pos % p.inner;
    int target = targets[pos];

    float lmax = -INFINITY;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lmax = max(lmax, logits[(o * p.classes + c) * p.inner + in]);
    }
    red[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lsum = 0.0f;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lsum += exp(logits[(o * p.classes + c) * p.inner + in] - maxv);
    }
    red[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float logit_t = logits[(o * p.classes + target) * p.inner + in];
        losses[pos] = -(logit_t - maxv - log(red[0]));
    }
}
kernel void gd_cross_entropy_reduce(device const float *losses           [[buffer(0)]],
                                    device float *out                    [[buffer(1)]],
                                    constant gd_metal_ce_params &p        [[buffer(2)]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float partial[GD_CE_TG];
    float local = 0.0f;
    for (int pos = (int)tid; pos < p.positions; pos += (int)tgsz) {
        local += losses[pos];
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = partial[0] / (float)p.positions;
    }
}
