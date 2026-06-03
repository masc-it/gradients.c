#include "metal_common.metal"

kernel void gd_cross_entropy_bwd(device const float *logits     [[buffer(0)]],
                                 device const int *targets      [[buffer(1)]],
                                 device const float *go_scalar  [[buffer(2)]],
                                 device float *dlogits          [[buffer(3)]],
                                 constant gd_metal_ce_params &p  [[buffer(4)]],
                                 device const float *valid_count [[buffer(5)]],
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
    float denom = p.has_ignore_index != 0 ? valid_count[0] : (float)p.positions;
    float scale = denom > 0.0f ? go_scalar[0] / denom : 0.0f;

    if (ignored || target < 0 || target >= p.classes || denom <= 0.0f) {
        for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
            dlogits[(o * p.classes + c) * p.inner + in] = 0.0f;
        }
        return;
    }

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
    float sum = red[0];

    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        float pc = exp(logits[(o * p.classes + c) * p.inner + in] - maxv) / sum;
        float onehot = (c == target) ? 1.0f : 0.0f;
        dlogits[(o * p.classes + c) * p.inner + in] = scale * (pc - onehot);
    }
}
