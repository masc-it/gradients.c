#include "metal_common.metal"

kernel void gd_rms_norm_wbwd(device const uchar *x               [[buffer(0)]],
                             device const uchar *go              [[buffer(1)]],
                             device float *partial               [[buffer(2)]],
                             constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                             uint tgid  [[threadgroup_position_in_grid]],
                             uint tid    [[thread_index_in_threadgroup]],
                             uint tgsz   [[threads_per_threadgroup]])
{
    threadgroup float inv_sh[GD_RMS_TG];
    int row_blocks = (p.rows + (int)tgsz - 1) / (int)tgsz;
    int rb = (int)tgid % row_blocks;
    int cb = (int)tgid / row_blocks;
    int c = cb * (int)tgsz + (int)tid;
    int row0 = rb * (int)tgsz;
    int tile = p.rows - row0;
    float acc = 0.0f;
    if (tile > (int)tgsz) {
        tile = (int)tgsz;
    }

    if ((int)tid < tile) {
        int rbase = (row0 + (int)tid) * p.last;
        float ss = 0.0f;
        for (int cc = 0; cc < p.last; ++cc) {
            float v = gd_load_float(x, p.dtype, (uint)(rbase + cc));
            ss += v * v;
        }
        inv_sh[tid] = 1.0f / sqrt(ss / (float)p.last + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < p.last) {
        for (int i = 0; i < tile; ++i) {
            int rbase = (row0 + i) * p.last;
            float g = gd_load_float(go, p.dtype, (uint)(rbase + c));
            float xv = gd_load_float(x, p.dtype, (uint)(rbase + c));
            acc += g * xv * inv_sh[i];
        }
        partial[rb * p.last + c] = acc;
    }
}
kernel void gd_rms_norm_wbwd_reduce(device const float *partial      [[buffer(0)]],
                                    device float *dweight            [[buffer(1)]],
                                    constant gd_metal_rmsnorm_params &p [[buffer(2)]],
                                    uint gid                         [[thread_position_in_grid]])
{
    int c = (int)gid;
    if (c >= p.last) {
        return;
    }
    int row_blocks = (p.rows + GD_RMS_TG - 1) / GD_RMS_TG;
    float acc = 0.0f;
    for (int rb = 0; rb < row_blocks; ++rb) {
        acc += partial[rb * p.last + c];
    }
    dweight[c] = acc;
}
