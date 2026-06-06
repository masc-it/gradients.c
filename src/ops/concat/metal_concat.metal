#include <metal_stdlib>
#include "metal_concat_types.h"

using namespace metal;

#define GD_CONCAT_DEFINE_TO_FULL_KERNEL(SUFFIX, TYPE)                                      \
kernel void gd_concat_to_full_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                device uchar *dst_bytes [[buffer(1)]],       \
                                                constant gd_metal_concat_args &args [[buffer(2)]], \
                                                uint gid [[thread_position_in_grid]])       \
{                                                                                            \
    ulong i = ulong(gid);                                                                    \
    if (i >= args.count) {                                                                   \
        return;                                                                              \
    }                                                                                        \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);         \
    ulong slice_block = args.slice_axis * args.inner;                                        \
    ulong outer = i / slice_block;                                                          \
    ulong rem = i - outer * slice_block;                                                     \
    ulong dst_index = outer * args.full_axis * args.inner + args.axis_offset * args.inner + rem; \
    dst[dst_index] = src[i];                                                                 \
}

#define GD_CONCAT_DEFINE_FROM_FULL_KERNEL(SUFFIX, TYPE)                                      \
kernel void gd_concat_from_full_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                  device uchar *dst_bytes [[buffer(1)]],     \
                                                  constant gd_metal_concat_args &args [[buffer(2)]], \
                                                  uint gid [[thread_position_in_grid]])     \
{                                                                                            \
    ulong i = ulong(gid);                                                                    \
    if (i >= args.count) {                                                                   \
        return;                                                                              \
    }                                                                                        \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);         \
    ulong slice_block = args.slice_axis * args.inner;                                        \
    ulong outer = i / slice_block;                                                          \
    ulong rem = i - outer * slice_block;                                                     \
    ulong src_index = outer * args.full_axis * args.inner + args.axis_offset * args.inner + rem; \
    dst[i] = src[src_index];                                                                 \
}

GD_CONCAT_DEFINE_TO_FULL_KERNEL(u8, uchar)
GD_CONCAT_DEFINE_TO_FULL_KERNEL(u16, ushort)
GD_CONCAT_DEFINE_TO_FULL_KERNEL(u32, uint)

GD_CONCAT_DEFINE_FROM_FULL_KERNEL(u8, uchar)
GD_CONCAT_DEFINE_FROM_FULL_KERNEL(u16, ushort)
GD_CONCAT_DEFINE_FROM_FULL_KERNEL(u32, uint)
