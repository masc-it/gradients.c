#include "metal_common.metal"

kernel void gd_cross_entropy(device const uchar *logits     [[buffer(0)]],
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
    bool ignored = (p.has_ignore_index != 0 && target == p.ignore_index);
    if (ignored) {
        if (tid == 0) {
            losses[pos] = 0.0f;
        }
        return;
    }
    if (target < 0 || target >= p.classes) {
        if (tid == 0) {
            losses[pos] = INFINITY;
        }
        return;
    }

    float lmax = -INFINITY;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        uint idx = (uint)((o * p.classes + c) * p.inner + in);
        lmax = max(lmax, gd_load_float(logits, p.dtype, idx));
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
        uint idx = (uint)((o * p.classes + c) * p.inner + in);
        lsum += exp(gd_load_float(logits, p.dtype, idx) - maxv);
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
        uint tidx = (uint)((o * p.classes + target) * p.inner + in);
        float logit_t = gd_load_float(logits, p.dtype, tidx);
        losses[pos] = -(logit_t - maxv - log(red[0]));
    }
}

kernel void gd_cross_entropy_reduce(device const float *losses           [[buffer(0)]],
                                    device const int *targets            [[buffer(1)]],
                                    device float *out                    [[buffer(2)]],
                                    constant gd_metal_ce_params &p        [[buffer(3)]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float partial[GD_CE_TG];
    threadgroup int counts[GD_CE_TG];
    float local = 0.0f;
    int count = 0;
    for (int pos = (int)tid; pos < p.positions; pos += (int)tgsz) {
        int target = targets[pos];
        if (p.has_ignore_index == 0 || target != p.ignore_index) {
            local += losses[pos];
            count += 1;
        }
    }
    partial[tid] = local;
    counts[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
            counts[tid] += counts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = counts[0] > 0 ? partial[0] / (float)counts[0] : 0.0f;
    }
}

kernel void gd_cross_entropy_count_valid(device const int *targets       [[buffer(0)]],
                                         device float *out               [[buffer(1)]],
                                         constant gd_metal_ce_params &p   [[buffer(2)]],
                                         uint tid  [[thread_index_in_threadgroup]],
                                         uint tgsz [[threads_per_threadgroup]])
{
    threadgroup int counts[GD_CE_TG];
    int count = 0;
    for (int pos = (int)tid; pos < p.positions; pos += (int)tgsz) {
        int target = targets[pos];
        if (p.has_ignore_index == 0 || target != p.ignore_index) {
            count += 1;
        }
    }
    counts[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            counts[tid] += counts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = (float)counts[0];
    }
}
