#include "metal_common.metal"

kernel void gd_rms_norm_qkv_norm(device const uchar *x                         [[buffer(0)]],
                                 device const uchar *weight                    [[buffer(1)]],
                                 device uchar *norm_out                        [[buffer(2)]],
                                 constant gd_metal_rms_norm_qkv_params &p      [[buffer(3)]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint tgsz [[threads_per_threadgroup]])
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
        gd_store_float(norm_out, p.dtype, (uint)(base + c), xv * inv * wv);
    }
}

static inline float gd_rms_norm_qkv_direct_normed(device const uchar *x,
                                                   device const uchar *weight,
                                                   constant gd_metal_rms_norm_qkv_params &p,
                                                   int row,
                                                   int col)
{
    int base = row * p.last;
    float sumsq = 0.0f;
    for (int c = 0; c < p.last; ++c) {
        float v = gd_load_float(x, p.dtype, (uint)(base + c));
        sumsq += v * v;
    }
    float inv = 1.0f / sqrt(sumsq / (float)p.last + p.eps);
    return gd_load_float(x, p.dtype, (uint)(base + col)) * inv *
           gd_load_float(weight, p.dtype, (uint)col);
}

static inline float gd_rms_norm_qkv_project(device const uchar *x,
                                             device const uchar *weight,
                                             device const uchar *w,
                                             constant gd_metal_rms_norm_qkv_params &p,
                                             int row,
                                             int out_col,
                                             int out_cols)
{
    int base = row * p.last;
    float sumsq = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < p.last; ++c) {
        float xv = gd_load_float(x, p.dtype, (uint)(base + c));
        sumsq += xv * xv;
    }
    float inv = 1.0f / sqrt(sumsq / (float)p.last + p.eps);
    for (int c = 0; c < p.last; ++c) {
        float xv = gd_load_float(x, p.dtype, (uint)(base + c));
        float wnorm = gd_load_float(weight, p.dtype, (uint)c);
        float wv = gd_load_float(w, p.dtype, (uint)(c * out_cols + out_col));
        acc += xv * inv * wnorm * wv;
    }
    return acc;
}

kernel void gd_rms_norm_qkv(device const uchar *x                         [[buffer(0)]],
                            device const uchar *weight                    [[buffer(1)]],
                            device const uchar *wq                        [[buffer(2)]],
                            device const uchar *wk                        [[buffer(3)]],
                            device const uchar *wv                        [[buffer(4)]],
                            device uchar *norm_out                        [[buffer(5)]],
                            device uchar *q_out                           [[buffer(6)]],
                            device uchar *k_out                           [[buffer(7)]],
                            device uchar *v_out                           [[buffer(8)]],
                            constant gd_metal_rms_norm_qkv_params &p      [[buffer(9)]],
                            uint gid [[thread_position_in_grid]])
{
    int row_width = p.last + p.q_cols + p.k_cols + p.v_cols;
    int lin = (int)gid;
    int row = row_width > 0 ? lin / row_width : 0;
    int col = row_width > 0 ? lin - row * row_width : 0;

    if (row >= p.rows) {
        return;
    }
    if (col < p.last) {
        float nv = gd_rms_norm_qkv_direct_normed(x, weight, p, row, col);
        gd_store_float(norm_out, p.dtype, (uint)(row * p.last + col), nv);
        return;
    }
    col -= p.last;
    if (col < p.q_cols) {
        float qv = gd_rms_norm_qkv_project(x, weight, wq, p, row, col, p.q_cols);
        gd_store_float(q_out, p.dtype, (uint)(row * p.q_cols + col), qv);
        return;
    }
    col -= p.q_cols;
    if (col < p.k_cols) {
        float kv = gd_rms_norm_qkv_project(x, weight, wk, p, row, col, p.k_cols);
        gd_store_float(k_out, p.dtype, (uint)(row * p.k_cols + col), kv);
        return;
    }
    col -= p.k_cols;
    if (col < p.v_cols) {
        float vv = gd_rms_norm_qkv_project(x, weight, wv, p, row, col, p.v_cols);
        gd_store_float(v_out, p.dtype, (uint)(row * p.v_cols + col), vv);
    }
}
