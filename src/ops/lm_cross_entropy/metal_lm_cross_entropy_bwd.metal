#include "metal_common.metal"

kernel void gd_lmce_dlogits_chunk(device float *logits               [[buffer(0)]],
                                  device const int *targets          [[buffer(1)]],
                                  device const float *go_scalar      [[buffer(2)]],
                                  device const float *m              [[buffer(3)]],
                                  device const float *l              [[buffer(4)]],
                                  constant gd_metal_lmce_params &p    [[buffer(5)]],
                                  uint gid                           [[thread_position_in_grid]])
{
    int idx = (int)gid;
    int total = p.rows * p.chunk_size;
    if (idx >= total) {
        return;
    }
    int row = idx / p.chunk_size;
    int c = idx - row * p.chunk_size;
    int cls = p.chunk_start + c;
    float prob = exp(logits[idx] - m[row]) / l[row];
    float onehot = (cls == targets[row]) ? 1.0f : 0.0f;
    logits[idx] = (go_scalar[0] / (float)p.rows) * (prob - onehot);
}
