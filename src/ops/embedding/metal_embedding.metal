#include <metal_stdlib>
#include "metal_embedding_types.h"

using namespace metal;

static inline float gd_embedding_nan(void)
{
    return as_type<float>(0x7fc00000u);
}

kernel void gd_embedding_forward_f16_kernel(device const uchar *tablebuf [[buffer(0)]],
                                            device const uchar *idsbuf [[buffer(1)]],
                                            device uchar *outbuf [[buffer(2)]],
                                            constant gd_metal_embedding_args &args [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    const ulong c = ulong(gid.x);
    const ulong row = ulong(gid.y);
    if (c >= args.dim || row >= args.ids_count) {
        return;
    }
    device const half *table = reinterpret_cast<device const half *>(tablebuf + args.table_offset);
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    const int id = ids[row];
    const ulong out_idx = row * args.dim + c;
    if (id < 0 || ulong(id) >= args.vocab) {
        out[out_idx] = half(gd_embedding_nan());
        return;
    }
    out[out_idx] = table[ulong(id) * args.dim + c];
}

kernel void gd_embedding_forward_f32_kernel(device const uchar *tablebuf [[buffer(0)]],
                                            device const uchar *idsbuf [[buffer(1)]],
                                            device uchar *outbuf [[buffer(2)]],
                                            constant gd_metal_embedding_args &args [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    const ulong c = ulong(gid.x);
    const ulong row = ulong(gid.y);
    if (c >= args.dim || row >= args.ids_count) {
        return;
    }
    device const float *table = reinterpret_cast<device const float *>(tablebuf + args.table_offset);
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    const int id = ids[row];
    const ulong out_idx = row * args.dim + c;
    if (id < 0 || ulong(id) >= args.vocab) {
        out[out_idx] = gd_embedding_nan();
        return;
    }
    out[out_idx] = table[ulong(id) * args.dim + c];
}

kernel void gd_embedding_forward_vec16_f16_kernel(device const uchar *tablebuf [[buffer(0)]],
                                                  device const uchar *idsbuf [[buffer(1)]],
                                                  device uchar *outbuf [[buffer(2)]],
                                                  constant gd_metal_embedding_args &args [[buffer(3)]],
                                                  uint2 gid [[thread_position_in_grid]])
{
    const ulong vec = ulong(gid.x);
    const ulong row = ulong(gid.y);
    const ulong dim_bytes = args.dim * 2ul;
    const ulong vecs = dim_bytes >> 4;
    if (vec >= vecs || row >= args.ids_count) {
        return;
    }
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    const int id = ids[row];
    const ulong out_byte = args.out_offset + row * dim_bytes + vec * 16ul;
    device uint4 *dst = reinterpret_cast<device uint4 *>(outbuf + out_byte);
    if (id < 0 || ulong(id) >= args.vocab) {
        dst[0] = uint4(0x7e007e00u);
        return;
    }
    const ulong src_byte = args.table_offset + ulong(id) * dim_bytes + vec * 16ul;
    device const uint4 *src = reinterpret_cast<device const uint4 *>(tablebuf + src_byte);
    dst[0] = src[0];
}

kernel void gd_embedding_forward_vec16_f32_kernel(device const uchar *tablebuf [[buffer(0)]],
                                                  device const uchar *idsbuf [[buffer(1)]],
                                                  device uchar *outbuf [[buffer(2)]],
                                                  constant gd_metal_embedding_args &args [[buffer(3)]],
                                                  uint2 gid [[thread_position_in_grid]])
{
    const ulong vec = ulong(gid.x);
    const ulong row = ulong(gid.y);
    const ulong dim_bytes = args.dim * 4ul;
    const ulong vecs = dim_bytes >> 4;
    if (vec >= vecs || row >= args.ids_count) {
        return;
    }
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    const int id = ids[row];
    const ulong out_byte = args.out_offset + row * dim_bytes + vec * 16ul;
    device uint4 *dst = reinterpret_cast<device uint4 *>(outbuf + out_byte);
    if (id < 0 || ulong(id) >= args.vocab) {
        dst[0] = uint4(0x7fc00000u);
        return;
    }
    const ulong src_byte = args.table_offset + ulong(id) * dim_bytes + vec * 16ul;
    device const uint4 *src = reinterpret_cast<device const uint4 *>(tablebuf + src_byte);
    dst[0] = src[0];
}

kernel void gd_embedding_zero_f32_kernel(device uchar *targetbuf [[buffer(0)]],
                                         constant gd_metal_embedding_args &args [[buffer(1)]],
                                         uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    const ulong total = args.vocab * args.dim;
    if (i >= total) {
        return;
    }
    device float *target = reinterpret_cast<device float *>(targetbuf + args.scratch_offset);
    target[i] = 0.0f;
}

kernel void gd_embedding_backward_scatter_f16_kernel(device const uchar *gradbuf [[buffer(0)]],
                                                     device const uchar *idsbuf [[buffer(1)]],
                                                     device uchar *targetbuf [[buffer(2)]],
                                                     constant gd_metal_embedding_args &args [[buffer(3)]],
                                                     uint2 gid [[thread_position_in_grid]])
{
    const ulong c = ulong(gid.x);
    const ulong row = ulong(gid.y);
    if (c >= args.dim || row >= args.ids_count) {
        return;
    }
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_out_offset);
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    device atomic_float *dtable = reinterpret_cast<device atomic_float *>(targetbuf + args.scratch_offset);
    const int id = ids[row];
    if (id < 0 || ulong(id) >= args.vocab) {
        return;
    }
    const float value = float(grad[row * args.dim + c]);
    atomic_fetch_add_explicit(dtable + ulong(id) * args.dim + c, value, memory_order_relaxed);
}

kernel void gd_embedding_backward_scatter_f32_kernel(device const uchar *gradbuf [[buffer(0)]],
                                                     device const uchar *idsbuf [[buffer(1)]],
                                                     device uchar *targetbuf [[buffer(2)]],
                                                     constant gd_metal_embedding_args &args [[buffer(3)]],
                                                     uint2 gid [[thread_position_in_grid]])
{
    const ulong c = ulong(gid.x);
    const ulong row = ulong(gid.y);
    if (c >= args.dim || row >= args.ids_count) {
        return;
    }
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device const int *ids = reinterpret_cast<device const int *>(idsbuf + args.ids_offset);
    device atomic_float *dtable = reinterpret_cast<device atomic_float *>(targetbuf + args.scratch_offset);
    const int id = ids[row];
    if (id < 0 || ulong(id) >= args.vocab) {
        return;
    }
    const float value = grad[row * args.dim + c];
    atomic_fetch_add_explicit(dtable + ulong(id) * args.dim + c, value, memory_order_relaxed);
}

kernel void gd_embedding_cast_f32_to_f16_kernel(device const uchar *scratchbuf [[buffer(0)]],
                                                device uchar *outbuf [[buffer(1)]],
                                                constant gd_metal_embedding_args &args [[buffer(2)]],
                                                uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    const ulong total = args.vocab * args.dim;
    if (i >= total) {
        return;
    }
    device const float *src = reinterpret_cast<device const float *>(scratchbuf + args.scratch_offset);
    device half *dst = reinterpret_cast<device half *>(outbuf + args.grad_table_offset);
    dst[i] = half(src[i]);
}
