#include "metal_common.metal"

kernel void gd_lmce_fwd_chunk(device const float *logits             [[buffer(0)]],
                              device const int *targets             [[buffer(1)]],
                              device float *m_out                   [[buffer(2)]],
                              device float *l_out                   [[buffer(3)]],
                              device float *target_logit            [[buffer(4)]],
                              constant gd_metal_lmce_params &p       [[buffer(5)]],
                              uint gid                              [[thread_position_in_grid]])
{
    int row = (int)gid;
    if (row >= p.rows) {
        return;
    }
    float old_m = p.first_chunk ? -INFINITY : m_out[row];
    float old_l = p.first_chunk ? 0.0f : l_out[row];
    float cm = -INFINITY;
    int base = row * p.chunk_size;
    for (int c = 0; c < p.chunk_size; ++c) {
        cm = max(cm, logits[base + c]);
    }
    float mnew = max(old_m, cm);
    float lnew = old_l * exp(old_m - mnew);
    for (int c = 0; c < p.chunk_size; ++c) {
        lnew += exp(logits[base + c] - mnew);
    }
    int t = targets[row];
    if (p.has_ignore_index == 0 || t != p.ignore_index) {
        if (t >= p.chunk_start && t < p.chunk_start + p.chunk_size) {
            target_logit[row] = logits[base + (t - p.chunk_start)];
        }
    }
    m_out[row] = mnew;
    l_out[row] = lnew;
}

kernel void gd_lmce_loss_rows(device const float *m                  [[buffer(0)]],
                              device const float *l                  [[buffer(1)]],
                              device const float *target_logit       [[buffer(2)]],
                              device float *losses                   [[buffer(3)]],
                              device const int *targets              [[buffer(4)]],
                              constant gd_metal_lmce_params &p       [[buffer(5)]],
                              uint gid                              [[thread_position_in_grid]])
{
    int row = (int)gid;
    if (row >= p.rows) {
        return;
    }
    int t = targets[row];
    if (p.has_ignore_index != 0 && t == p.ignore_index) {
        losses[row] = 0.0f;
        return;
    }
    if (t < 0 || t >= p.vocab) {
        losses[row] = INFINITY;
        return;
    }
    losses[row] = -(target_logit[row] - m[row] - log(l[row]));
}
